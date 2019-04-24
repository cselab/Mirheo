#include <mpi.h>
#include <cuda_runtime.h>

#include <core/bouncers/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/initial_conditions/uniform.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/logger.h>
#include <core/object_belonging/interface.h>
#include <core/postproc.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/utils/make_unique.h>
#include <core/version.h>
#include <core/walls/interface.h>
#include <core/walls/simple_stationary_wall.h>
#include <core/walls/wall_helpers.h>
#include <plugins/interface.h>

#include "ymero.h"

static void createCartComm(MPI_Comm comm, int3 nranks3D, MPI_Comm *cartComm)
{
    int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
    int periods[] = {1, 1, 1};
    int reorder = 1;
    MPI_Check(MPI_Cart_create(comm, 3, ranksArr, periods, reorder, cartComm));
}

/// Map intro-node ranks to different GPUs
/// https://stackoverflow.com/a/40122688/3535276
static void selectIntraNodeGPU(const MPI_Comm& source)
{
    MPI_Comm shmcomm;
    MPI_Check( MPI_Comm_split_type(source, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm) );
    
    int shmrank, shmsize;
    MPI_Check( MPI_Comm_rank(shmcomm, &shmrank) );
    MPI_Check( MPI_Comm_size(shmcomm, &shmsize) );

    info("Detected %d ranks per node, my intra-node ID will be %d", shmsize, shmrank);

    int ngpus;
    CUDA_Check( cudaGetDeviceCount(&ngpus) );
    
    int mygpu = shmrank % ngpus;

    info("Found %d GPUs per node, will use GPU %d", ngpus, mygpu);

    CUDA_Check( cudaSetDevice(mygpu) );
    CUDA_Check( cudaDeviceReset() );

    MPI_Check( MPI_Comm_free(&shmcomm) );
}

void YMeRo::init(int3 nranks3D, float3 globalDomainSize, float dt, std::string logFileName, int verbosity,
                 int checkpointEvery, std::string checkpointFolder, CheckpointIdAdvanceMode checkpointMode, bool gpuAwareMPI)
{
    int nranks;
    
    initLogger(comm, logFileName, verbosity);   

    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    MPI_Check( MPI_Comm_size(comm, &nranks) );
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    if      (nranks3D.x * nranks3D.y * nranks3D.z     == nranks) noPostprocess = true;
    else if (nranks3D.x * nranks3D.y * nranks3D.z * 2 == nranks) noPostprocess = false;
    else die("Asked for %d x %d x %d processes, but provided %d", nranks3D.x, nranks3D.y, nranks3D.z, nranks);

    if (rank == 0) sayHello();    

    if (noPostprocess) {
        warn("No postprocess will be started now, use this mode for debugging. All the joint plugins will be turned off too.");
        
        selectIntraNodeGPU(comm);

        createCartComm(comm, nranks3D, &cartComm);
        state = std::make_shared<YmrState> (createDomainInfo(cartComm, globalDomainSize), dt);
        sim = std::make_unique<Simulation> (cartComm, MPI_COMM_NULL, getState(),
                                            checkpointEvery, checkpointFolder, checkpointMode, gpuAwareMPI);
        computeTask = 0;
        return;
    }

    info("Program started, splitting communicator");

    MPI_Comm splitComm;
    
    computeTask = rank % 2;
    MPI_Check( MPI_Comm_split(comm, computeTask, rank, &splitComm) );

    if (isComputeTask())
    {
        MPI_Check( MPI_Comm_dup(splitComm, &compComm) );
        MPI_Check( MPI_Intercomm_create(compComm, 0, comm, 1, 0, &interComm) );

        MPI_Check( MPI_Comm_rank(compComm, &rank) );
        selectIntraNodeGPU(compComm);

        createCartComm(compComm, nranks3D, &cartComm);
        state = std::make_shared<YmrState> (createDomainInfo(cartComm, globalDomainSize), dt);
        sim = std::make_unique<Simulation> (cartComm, interComm, getState(),
                                            checkpointEvery, checkpointFolder, checkpointMode, gpuAwareMPI);
    }
    else
    {
        MPI_Check( MPI_Comm_dup(splitComm, &ioComm) );
        MPI_Check( MPI_Intercomm_create(ioComm,   0, comm, 0, 0, &interComm) );

        MPI_Check( MPI_Comm_rank(ioComm, &rank) );

        post = std::make_unique<Postprocess> (ioComm, interComm, checkpointFolder, checkpointMode);
    }

    MPI_Check( MPI_Comm_free(&splitComm) );
}

void YMeRo::initLogger(MPI_Comm comm, std::string logFileName, int verbosity)
{
    if      (logFileName == "stdout")  logger.init(comm, stdout,             verbosity);
    else if (logFileName == "stderr")  logger.init(comm, stderr,             verbosity);
    else                               logger.init(comm, logFileName+".log", verbosity);
}

YMeRo::YMeRo(PyTypes::int3 nranks3D, PyTypes::float3 globalDomainSize, float dt,
             std::string logFileName, int verbosity, int checkpointEvery, std::string checkpointFolder,
             CheckpointIdAdvanceMode checkpointMode, bool gpuAwareMPI, bool noSplash) :
    noSplash(noSplash)
{
    MPI_Init(nullptr, nullptr);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    initializedMpi = true;

    init( make_int3(nranks3D), make_float3(globalDomainSize), dt, logFileName, verbosity,
          checkpointEvery, checkpointFolder, checkpointMode, gpuAwareMPI);
}

YMeRo::YMeRo(long commAdress, PyTypes::int3 nranks3D, PyTypes::float3 globalDomainSize, float dt,
             std::string logFileName, int verbosity, int checkpointEvery, std::string checkpointFolder,
             CheckpointIdAdvanceMode checkpointMode, bool gpuAwareMPI, bool noSplash) :
    noSplash(noSplash)
{
    // see https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
    MPI_Comm comm = *((MPI_Comm*) commAdress);
    MPI_Comm_dup(comm, &this->comm);
    init( make_int3(nranks3D), make_float3(globalDomainSize), dt, logFileName, verbosity,
          checkpointEvery, checkpointFolder, checkpointMode, gpuAwareMPI);    
}

YMeRo::YMeRo(MPI_Comm comm, PyTypes::int3 nranks3D, PyTypes::float3 globalDomainSize, float dt,
             std::string logFileName, int verbosity, int checkpointEvery, std::string checkpointFolder,
             CheckpointIdAdvanceMode checkpointMode, bool gpuAwareMPI, bool noSplash) :
    noSplash(noSplash)
{
    MPI_Comm_dup(comm, &this->comm);
    init( make_int3(nranks3D), make_float3(globalDomainSize), dt, logFileName, verbosity,
          checkpointEvery, checkpointFolder, checkpointMode, gpuAwareMPI);
}

static void safeCommFree(MPI_Comm *comm)
{
    if (*comm != MPI_COMM_NULL)
        MPI_Check( MPI_Comm_free(comm) );
}

YMeRo::~YMeRo()
{
    debug("YMeRo coordinator is destroyed");
    
    sim.reset();
    post.reset();

    safeCommFree(&comm);
    safeCommFree(&cartComm);
    safeCommFree(&ioComm);
    safeCommFree(&compComm);
    safeCommFree(&interComm);
    
    if (initializedMpi)
        MPI_Finalize();
}

void YMeRo::registerParticleVector(const std::shared_ptr<ParticleVector>& pv, const std::shared_ptr<InitialConditions>& ic, int checkpointEvery)
{
    if (isComputeTask())
        sim->registerParticleVector(pv, ic, checkpointEvery);
}

void YMeRo::registerIntegrator(const std::shared_ptr<Integrator>& integrator)
{
    if (isComputeTask())
        sim->registerIntegrator(integrator);
}

void YMeRo::registerInteraction(const std::shared_ptr<Interaction>& interaction)
{
    if (isComputeTask())
        sim->registerInteraction(interaction);
}

void YMeRo::registerWall(const std::shared_ptr<Wall>& wall, int checkEvery)
{
    if (isComputeTask())
        sim->registerWall(wall, checkEvery);
}

void YMeRo::registerBouncer(const std::shared_ptr<Bouncer>& bouncer)
{
    if (isComputeTask())
        sim->registerBouncer(bouncer);
}

void YMeRo::registerObjectBelongingChecker (const std::shared_ptr<ObjectBelongingChecker>& checker, ObjectVector* ov)
{
    if (isComputeTask())
    {
        sim->registerObjectBelongingChecker(checker);
        sim->setObjectBelongingChecker(checker->name, ov->name);
    }
}

void YMeRo::registerPlugins(const std::shared_ptr<SimulationPlugin>& simPlugin, const std::shared_ptr<PostprocessPlugin>& postPlugin)
{
    if (isComputeTask())
    {
        if ( simPlugin != nullptr && !(simPlugin->needPostproc() && noPostprocess) )
            sim->registerPlugin(simPlugin);
    }
    else
    {
        if ( postPlugin != nullptr && !noPostprocess )
            post->registerPlugin(postPlugin);
    }
}

void YMeRo::setIntegrator(Integrator* integrator, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setIntegrator(integrator->name, pv->name);
}

void YMeRo::setInteraction(Interaction* interaction, ParticleVector* pv1, ParticleVector* pv2)
{
    if (isComputeTask())
        sim->setInteraction(interaction->name, pv1->name, pv2->name);
}

void YMeRo::setBouncer(Bouncer* bouncer, ObjectVector* ov, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setBouncer(bouncer->name, ov->name, pv->name);
}

void YMeRo::setWallBounce(Wall* wall, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setWallBounce(wall->name, pv->name);
}

YmrState* YMeRo::getState()
{
    return state.get();
}

const YmrState* YMeRo::getState() const
{
    return state.get();
}

std::shared_ptr<YmrState> YMeRo::getYmrState()
{
    return state;
}

void YMeRo::dumpWalls2XDMF(std::vector<std::shared_ptr<Wall>> walls, PyTypes::float3 h, std::string filename)
{
    if (!isComputeTask()) return;

    info("Dumping SDF into XDMF:\n");

    std::vector<SDF_basedWall*> sdfWalls;
    for (auto &wall : walls)
    {
        auto sdfWall = dynamic_cast<SDF_basedWall*>(wall.get());
        if (sdfWall == nullptr)
            die("Only sdf-based walls are supported!");        
        else
            sdfWalls.push_back(sdfWall);

        // Check if the wall is set up
        sim->getWallByNameOrDie(wall->name);
    }
    
    auto path = parentPath(filename);
    if (path != filename)
        createFoldersCollective(sim->cartComm, path);
    ::dumpWalls2XDMF(sdfWalls, make_float3(h), state->domain, filename, sim->cartComm);
}

double YMeRo::computeVolumeInsideWalls(std::vector<std::shared_ptr<Wall>> walls, long nSamplesPerRank)
{
    if (!isComputeTask()) return 0;

    info("Computing volume inside walls\n");
    
    std::vector<SDF_basedWall*> sdfWalls;
    for (auto &wall : walls)
    {
        auto sdfWall = dynamic_cast<SDF_basedWall*>(wall.get());
        if (sdfWall == nullptr)
            die("Only sdf-based walls are supported!");        
        else
            sdfWalls.push_back(sdfWall);

        // Check if the wall is set up
        sim->getWallByNameOrDie(wall->name);
    }

    return volumeInsideWalls(sdfWalls, state->domain, sim->cartComm, nSamplesPerRank);
}

std::shared_ptr<ParticleVector> YMeRo::makeFrozenWallParticles(std::string pvName,
                                                               std::vector<std::shared_ptr<Wall>> walls,
                                                               std::vector<std::shared_ptr<Interaction>> interactions,
                                                               std::shared_ptr<Integrator> integrator,
                                                               float density, int nsteps)
{
    if (!isComputeTask()) return nullptr;

    // Walls are not directly reusable in other simulations,
    // because they store some information like cell-lists
    //
    // But here we don't pass the wall into the other simulation,
    // we just use it to filter particles, which is totally fine
    
    info("Generating frozen particles for walls");

    std::vector<SDF_basedWall*> sdfWalls;

    for (auto &wall : walls) {
        auto sdfWall = dynamic_cast<SDF_basedWall*>(wall.get());
        if (sdfWall == nullptr)
            die("Only sdf-based walls are supported now!");        
        else
            sdfWalls.push_back(sdfWall);

        // Check if the wall is set up
        sim->getWallByNameOrDie(wall->name);

        info("Working with wall '%s'", wall->name.c_str());   
    }

    YmrState stateCpy = *getState();
    
    Simulation wallsim(sim->cartComm, MPI_COMM_NULL, getState());

    float mass = 1.0;
    auto pv = std::make_shared<ParticleVector>(getState(), pvName, mass);
    auto ic = std::make_shared<UniformIC>(density);
    
    wallsim.registerParticleVector(pv, ic, 0);
    
    wallsim.registerIntegrator(integrator);
    
    wallsim.setIntegrator (integrator->name,  pv->name);

    for (auto& interaction : interactions) {
        wallsim.registerInteraction(interaction);        
        wallsim.setInteraction(interaction->name, pv->name, pv->name);
    }
    
    wallsim.init();
    wallsim.run(nsteps);

    float effectiveCutoff = wallsim.getMaxEffectiveCutoff();
    
    const float wallThicknessTolerance = 0.2f;
    const float wallLevelSet = 0.0f;
    float wallThickness = effectiveCutoff + wallThicknessTolerance;

    info("wall thickness is set to %g", wallThickness);
    
    freezeParticlesInWalls(sdfWalls, pv.get(), wallLevelSet, wallLevelSet + wallThickness);
    info("\n");

    sim->registerParticleVector(pv, nullptr);

    for (auto &wall : walls)
        wall->attachFrozen(pv.get());

    // go back to initial state
    *state = stateCpy;
    
    return pv;
}

std::shared_ptr<ParticleVector> YMeRo::makeFrozenRigidParticles(std::shared_ptr<ObjectBelongingChecker> checker,
                                                                std::shared_ptr<ObjectVector> shape,
                                                                std::shared_ptr<InitialConditions> icShape,
                                                                std::vector<std::shared_ptr<Interaction>> interactions,
                                                                std::shared_ptr<Integrator>   integrator,
                                                                float density, int nsteps)
{
    if (!isComputeTask()) return nullptr;

    auto insideName = "inside_" + shape->name;
    
    info("Generating frozen particles for rigid object '%s'...\n\n", shape->name.c_str());

    if (shape->local()->nObjects > 1)
        die("expected no more than one object vector; given %d", shape->local()->nObjects);
    

    float mass = 1.0;
    auto pv = std::make_shared<ParticleVector>(getState(), "outside__" + shape->name, mass);
    auto ic = std::make_shared<UniformIC>(density);

    YmrState stateCpy = *getState();

    {
        Simulation eqsim(sim->cartComm, MPI_COMM_NULL, getState());
    
        eqsim.registerParticleVector(pv, ic, 0);

        eqsim.registerIntegrator(integrator);
        eqsim.setIntegrator (integrator->name,  pv->name);
        
        for (auto& interaction : interactions) {
            eqsim.registerInteraction(interaction);        
            eqsim.setInteraction(interaction->name, pv->name, pv->name);
        }               
    
        eqsim.init();
        eqsim.run(nsteps);
    }

    Simulation freezesim(sim->cartComm, MPI_COMM_NULL, getState());

    freezesim.registerParticleVector(pv, nullptr, 0);
    freezesim.registerParticleVector(shape, icShape, 0);
    freezesim.registerObjectBelongingChecker (checker);
    freezesim.setObjectBelongingChecker(checker->name, shape->name);
    freezesim.applyObjectBelongingChecker(checker->name, pv->name, insideName, pv->name, 0);

    freezesim.init();
    freezesim.run(1);

    // go back to initial state
    *state = stateCpy;

    return freezesim.getSharedPVbyName(insideName);
}

std::shared_ptr<ParticleVector> YMeRo::applyObjectBelongingChecker(ObjectBelongingChecker* checker,
                                                                      ParticleVector* pv,
                                                                      int checkEvery,
                                                                      std::string inside,
                                                                      std::string outside,
                                                                      int checkpointEvery)
{
    if (!isComputeTask()) return nullptr;
    
    if ( (inside != "" && outside != "") || (inside == "" && outside == "") )
        die("One and only one option can be specified for belonging checker '%s': inside or outside",
            checker->name.c_str());
    
    std::string newPVname;
    
    if (inside == "")
    {
        inside = pv->name;
        newPVname = outside;
    }
    if (outside == "")
    {
        outside = pv->name;
        newPVname = inside;
    }
        
    sim->applyObjectBelongingChecker(checker->name, pv->name, inside, outside, checkEvery, checkpointEvery);
    return sim->getSharedPVbyName(newPVname);
}

void YMeRo::sayHello()
{
    if (noSplash) return;
    
    static const int max_length_version =  9;
    static const int max_length_sha1    = 46;
    std::string version = Version::ymr_version;
    std::string sha1    = Version::git_SHA1;

    int missing_spaces = max(0, max_length_version - (int) version.size());
    version.append(missing_spaces, ' ');

    missing_spaces = max(0, max_length_sha1 - (int) sha1.size());
    sha1.append(missing_spaces, ' ');
    
    printf("\n");
    printf("**************************************************\n");
    printf("*                YMeRo %s                 *\n", version.c_str());
    printf("* %s *\n", sha1.c_str());
    printf("**************************************************\n");
    printf("\n");
}

void YMeRo::setup()
{
    if (initialized) return;
    
    if (isComputeTask())  sim->init();
    else                 post->init();
    
    initialized = true;
}

void YMeRo::restart(std::string folder)
{
    if (isComputeTask())
        sim->restart(folder);
    else
        post->restart(folder);
}

bool YMeRo::isComputeTask() const
{
    return (computeTask == 0);
}

bool YMeRo::isMasterTask() const
{
    return (rank == 0 && isComputeTask());
}

void YMeRo::saveDependencyGraph_GraphML(std::string fname, bool current) const
{
    if (isComputeTask())
        sim->saveDependencyGraph_GraphML(fname, current);
}

void YMeRo::startProfiler()
{
    if (isComputeTask())
        sim->startProfiler();
}

void YMeRo::stopProfiler()
{
    if (isComputeTask())
        sim->stopProfiler();
}

void YMeRo::run(int nsteps)
{
    setup();
    
    if (isComputeTask()) sim->run(nsteps);
    else                post->run();

    MPI_Check( MPI_Barrier(comm) );
}


