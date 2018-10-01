#include "udevicex.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <core/logger.h>
#include <core/simulation.h>
#include <core/postproc.h>
#include <plugins/interface.h>

#include <core/utils/make_unique.h>
#include <core/utils/folders.h>
#include <core/utils/cuda_common.h>
#include <cuda_runtime.h>

#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/walls/interface.h>
#include <core/bouncers/interface.h>
#include <core/object_belonging/interface.h>
#include <plugins/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

#include <core/walls/simple_stationary_wall.h>
#include <core/walls/wall_helpers.h>
#include <core/initial_conditions/uniform_ic.h>

#include "version.h"

void uDeviceX::init(int3 nranks3D, float3 globalDomainSize, std::string logFileName, int verbosity,
                    int checkpointEvery, std::string restartFolder, bool gpuAwareMPI)
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

    MPI_Comm ioComm, compComm, interComm, splitComm;

    if (noPostprocess) {
        warn("No postprocess will be started now, use this mode for debugging. All the joint plugins will be turned off too.");

        sim = std::make_unique<Simulation> (nranks3D, globalDomainSize,
                                            comm, MPI_COMM_NULL,
                                            checkpointEvery, restartFolder, gpuAwareMPI);
        computeTask = 0;
        return;
    }

    info("Program started, splitting communicator");

    computeTask = rank % 2;
    MPI_Check( MPI_Comm_split(comm, computeTask, rank, &splitComm) );

    if (isComputeTask())
    {
        MPI_Check( MPI_Comm_dup(splitComm, &compComm) );
        MPI_Check( MPI_Intercomm_create(compComm, 0, comm, 1, 0, &interComm) );

        MPI_Check( MPI_Comm_rank(compComm, &rank) );

        sim = std::make_unique<Simulation> (nranks3D, globalDomainSize,
                                            compComm, interComm,
                                            checkpointEvery, restartFolder, gpuAwareMPI);
    }
    else
    {
        MPI_Check( MPI_Comm_dup(splitComm, &ioComm) );
        MPI_Check( MPI_Intercomm_create(ioComm,   0, comm, 0, 0, &interComm) );

        MPI_Check( MPI_Comm_rank(ioComm, &rank) );

        post = std::make_unique<Postprocess> (ioComm, interComm);
    }
}

void uDeviceX::initLogger(MPI_Comm comm, std::string logFileName, int verbosity)
{
    if      (logFileName == "stdout")  logger.init(comm, stdout,             verbosity);
    else if (logFileName == "stderr")  logger.init(comm, stderr,             verbosity);
    else                               logger.init(comm, logFileName+".log", verbosity);
}

uDeviceX::uDeviceX(std::tuple<int, int, int> nranks3D, std::tuple<float, float, float> globalDomainSize,
                   std::string logFileName, int verbosity, int checkpointEvery, std::string restartFolder, bool gpuAwareMPI)
{
    MPI_Init(nullptr, nullptr);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    initializedMpi = true;

    init( make_int3(nranks3D), make_float3(globalDomainSize), logFileName, verbosity, checkpointEvery, restartFolder, gpuAwareMPI);
}

uDeviceX::uDeviceX(long commAdress, PyTypes::int3 nranks3D, PyTypes::float3 globalDomainSize,
                   std::string logFileName, int verbosity,
                   int checkpointEvery, std::string restartFolder, bool gpuAwareMPI)
{
    // see https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
    MPI_Comm comm = *((MPI_Comm*) commAdress);
    MPI_Comm_dup(comm, &this->comm);
    init( make_int3(nranks3D), make_float3(globalDomainSize), logFileName, verbosity, checkpointEvery, restartFolder, gpuAwareMPI);    
}

uDeviceX::uDeviceX(MPI_Comm comm, std::tuple<int, int, int> nranks3D, std::tuple<float, float, float> globalDomainSize,
                   std::string logFileName, int verbosity, int checkpointEvery, std::string restartFolder, bool gpuAwareMPI)
{
    MPI_Comm_dup(comm, &this->comm);
    init( make_int3(nranks3D), make_float3(globalDomainSize), logFileName, verbosity, checkpointEvery, restartFolder, gpuAwareMPI);
}

uDeviceX::~uDeviceX()
{
    debug("uDeviceX coordinator is destroyed");
    
    sim.release();
    post.release();

    if (initializedMpi)
        MPI_Finalize();
}



void uDeviceX::registerParticleVector(const std::shared_ptr<ParticleVector>& pv, const std::shared_ptr<InitialConditions>& ic, int checkpointEvery)
{
    if (isComputeTask())
        sim->registerParticleVector(pv, ic, checkpointEvery);
}
void uDeviceX::registerIntegrator(const std::shared_ptr<Integrator>& integrator)
{
    if (isComputeTask())
        sim->registerIntegrator(integrator);
}
void uDeviceX::registerInteraction(const std::shared_ptr<Interaction>& interaction)
{
    if (isComputeTask())
        sim->registerInteraction(interaction);
}
void uDeviceX::registerWall(const std::shared_ptr<Wall>& wall, int checkEvery)
{
    if (isComputeTask())
        sim->registerWall(wall, checkEvery);
}
void uDeviceX::registerBouncer(const std::shared_ptr<Bouncer>& bouncer)
{
    if (isComputeTask())
        sim->registerBouncer(bouncer);
}
void uDeviceX::registerObjectBelongingChecker (const std::shared_ptr<ObjectBelongingChecker>& checker, ObjectVector* ov)
{
    if (isComputeTask())
    {
        sim->registerObjectBelongingChecker(checker);
        sim->setObjectBelongingChecker(checker->name, ov->name);
    }
}
void uDeviceX::registerPlugins(const std::shared_ptr<SimulationPlugin>& simPlugin, const std::shared_ptr<PostprocessPlugin>& postPlugin)
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

void uDeviceX::setIntegrator(Integrator* integrator, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setIntegrator(integrator->name, pv->name);
}
void uDeviceX::setInteraction(Interaction* interaction, ParticleVector* pv1, ParticleVector* pv2)
{
    if (isComputeTask())
        sim->setInteraction(interaction->name, pv1->name, pv2->name);
}
void uDeviceX::setBouncer(Bouncer* bouncer, ObjectVector* ov, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setBouncer(bouncer->name, ov->name, pv->name);
}
void uDeviceX::setWallBounce(Wall* wall, ParticleVector* pv)
{
    if (isComputeTask())
        sim->setWallBounce(wall->name, pv->name);
}

void uDeviceX::dumpWalls2XDMF(std::vector<std::shared_ptr<Wall>> walls, PyTypes::float3 h, std::string filename)
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
    ::dumpWalls2XDMF(sdfWalls, make_float3(h), sim->domain, filename, sim->cartComm);
}

std::shared_ptr<ParticleVector> uDeviceX::makeFrozenWallParticles(std::string pvName,
                                                                  std::vector<std::shared_ptr<Wall>> walls,
                                                                  std::shared_ptr<Interaction> interaction,
                                                                  std::shared_ptr<Integrator>   integrator,
                                                                  float density, int nsteps)
{
    if (!isComputeTask()) return nullptr;

    // Walls are not directly reusable in other simulations,
    // because they store some information like cell-lists
    //
    // But here we don't pass the wall into the other simulation,
    // we just use it to filter particles, which is totally fine
    
    info("Generating frozen particles for walls:\n");

    std::vector<SDF_basedWall*> sdfWalls;

    for (auto &wall : walls) {
        auto sdfWall = dynamic_cast<SDF_basedWall*>(wall.get());
        if (sdfWall == nullptr)
            die("Only sdf-based walls are supported now!");        
        else
            sdfWalls.push_back(sdfWall);

        // Check if the wall is set up
        sim->getWallByNameOrDie(wall->name);

        info("\t%s", wall->name.c_str());
    }
    info("\n\n");
    
    Simulation wallsim(sim->nranks3D, sim->domain.globalSize, sim->cartComm, MPI_COMM_NULL, false);

    float mass = 1.0;
    auto pv = std::make_shared<ParticleVector>(pvName, mass);
    auto ic = std::make_shared<UniformIC>(density);
    
    wallsim.registerParticleVector(pv, ic, 0);
    wallsim.registerInteraction(interaction);

    wallsim.registerIntegrator(integrator);
    
    wallsim.setInteraction(interaction->name, pv->name, pv->name);
    wallsim.setIntegrator (integrator->name,  pv->name);
    
    wallsim.init();
    wallsim.run(nsteps);
    
    freezeParticlesInWalls(sdfWalls, pv.get(), 0.0f, interaction->rc + 0.2f);
    
    sim->registerParticleVector(pv, nullptr);

    for (auto &wall : walls)
        wall->attachFrozen(pv.get());
    
    return pv;
}

std::shared_ptr<ParticleVector> uDeviceX::makeFrozenRigidParticles(std::shared_ptr<ObjectBelongingChecker> checker,
                                                                   std::shared_ptr<ObjectVector> shape,
                                                                   std::shared_ptr<InitialConditions> icShape,
                                                                   std::shared_ptr<Interaction> interaction,
                                                                   std::shared_ptr<Integrator>   integrator,
                                                                   float density, int nsteps)
{
    if (!isComputeTask()) return nullptr;

    auto insideName = "inside_" + shape->name;
    
    info("Generating frozen particles for rigid object '%s'...\n\n", shape->name.c_str());

    if (shape->local()->nObjects > 1)
        die("expected no more than one object vector; given %d", shape->local()->nObjects);
    
    
    auto pv = std::make_shared<ParticleVector>("outside__" + shape->name, 1.0);
    auto ic = std::make_shared<UniformIC>(density);

    {
        Simulation eqsim(sim->nranks3D, sim->domain.globalSize, sim->cartComm, MPI_COMM_NULL, false);
    
        eqsim.registerParticleVector(pv, ic, 0);
        eqsim.registerInteraction(interaction);
        eqsim.registerIntegrator(integrator);
    
        eqsim.setInteraction(interaction->name, pv->name, pv->name);
        eqsim.setIntegrator (integrator->name,  pv->name);
    
        eqsim.init();
        eqsim.run(nsteps);
    }
    
    Simulation freezesim(sim->nranks3D, sim->domain.globalSize, sim->cartComm, MPI_COMM_NULL, false);

    freezesim.registerParticleVector(pv, nullptr, 0);
    freezesim.registerParticleVector(shape, icShape, 0);
    freezesim.registerObjectBelongingChecker (checker);
    freezesim.setObjectBelongingChecker(checker->name, shape->name);
    freezesim.applyObjectBelongingChecker(checker->name, pv->name, insideName, pv->name, 0);

    freezesim.init();
    freezesim.run(1);

    return freezesim.getSharedPVbyName(insideName);
}


std::shared_ptr<ParticleVector> uDeviceX::applyObjectBelongingChecker(ObjectBelongingChecker* checker,
                                                                      ParticleVector* pv,
                                                                      int checkEvery,
                                                                      std::string inside,
                                                                      std::string outside)
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
        
    sim->applyObjectBelongingChecker(checker->name, pv->name, inside, outside, checkEvery);
    return sim->getSharedPVbyName(newPVname);
}

void uDeviceX::sayHello()
{
    static const int max_length_version =  9;
    static const int max_length_sha1    = 46;
    std::string version = Version::udx_version;
    std::string sha1    = Version::git_SHA1;

    int missing_spaces = max(0, max_length_version - (int) version.size());
    version.append(missing_spaces, ' ');

    missing_spaces = max(0, max_length_sha1 - (int) sha1.size());
    sha1.append(missing_spaces, ' ');
    
    printf("\n");
    printf("**************************************************\n");
    printf("*              uDeviceX %s                *\n", version.c_str());
    printf("* %s *\n", sha1.c_str());
    printf("**************************************************\n");
    printf("\n");
}

bool uDeviceX::isComputeTask()
{
    return (computeTask == 0);
}

bool uDeviceX::isMasterTask()
{
    return (rank == 0 && isComputeTask());
}


void uDeviceX::run(int nsteps)
{
    if (isComputeTask())
    {
        if (!initialized)
        {
            sim->init();
            initialized = true;
        }
        sim->run(nsteps);
    }
    else
    {
        if (!initialized)
        {
            post->init();
            initialized = true;
        }
        post->run();
    }
    
    MPI_Check( MPI_Barrier(comm) );
}

