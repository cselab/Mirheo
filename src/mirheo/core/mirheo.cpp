#include "mirheo.h"

#include <mirheo/core/bouncers/interface.h>
#include <mirheo/core/initial_conditions/interface.h>
#include <mirheo/core/initial_conditions/uniform.h>
#include <mirheo/core/integrators/interface.h>
#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/object_belonging/interface.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/postproc.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/version.h>
#include <mirheo/core/walls/interface.h>
#include <mirheo/core/walls/simple_stationary_wall.h>
#include <mirheo/core/walls/wall_helpers.h>

#include <cuda_runtime.h>
#include <memory>
#include <mpi.h>

namespace mirheo
{

LogInfo::LogInfo(const std::string& fileName, int verbosityLvl, bool noSplash) :
    fileName(fileName),
    verbosityLvl(verbosityLvl),
    noSplash(noSplash)
{}
MIRHEO_MEMBER_VARS_3(LogInfo, fileName, verbosityLvl, noSplash);

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

void Mirheo::init(int3 nranks3D, real3 globalDomainSize, real dt, LogInfo logInfo,
                  CheckpointInfo checkpointInfo, bool gpuAwareMPI)
{
    int nranks;

    initLogger(comm_, logInfo);   

    MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);

    MPI_Check( MPI_Comm_size(comm_, &nranks) );
    MPI_Check( MPI_Comm_rank(comm_, &rank_) );

    if      (nranks3D.x * nranks3D.y * nranks3D.z     == nranks) noPostprocess_ = true;
    else if (nranks3D.x * nranks3D.y * nranks3D.z * 2 == nranks) noPostprocess_ = false;
    else die("Asked for %d x %d x %d processes, but provided %d", nranks3D.x, nranks3D.y, nranks3D.z, nranks);

    if (rank_ == 0 && !logInfo.noSplash)
        sayHello();

    checkpointInfo.folder = makePath(checkpointInfo.folder);
    
    if (noPostprocess_)
    {
        warn("No postprocess will be started now, use this mode for debugging. All the joint plugins will be turned off too.");
        
        selectIntraNodeGPU(comm_);

        createCartComm(comm_, nranks3D, &cartComm_);
        state_ = std::make_shared<MirState> (createDomainInfo(cartComm_, globalDomainSize), dt);
        sim_ = std::make_unique<Simulation> (cartComm_, MPI_COMM_NULL, getState(),
                                            checkpointInfo, gpuAwareMPI);
        computeTask_ = 0;
        return;
    }

    info("Program started, splitting communicator");

    MPI_Comm splitComm;
    
    // Note: Update `is*Task()` functions if modifying this.
    computeTask_ = rank_ % 2;
    MPI_Check( MPI_Comm_split(comm_, computeTask_, rank_, &splitComm) );

    const int localLeader  = 0;
    const int remoteLeader = isComputeTask() ? 1 : 0;
    const int tag = 42;

    if (isComputeTask())
    {
        MPI_Check( MPI_Comm_dup(splitComm, &compComm_) );
        MPI_Check( MPI_Intercomm_create(compComm_, localLeader, comm_, remoteLeader, tag, &interComm_) );

        MPI_Check( MPI_Comm_rank(compComm_, &rank_) );
        selectIntraNodeGPU(compComm_);

        createCartComm(compComm_, nranks3D, &cartComm_);
        state_ = std::make_shared<MirState> (createDomainInfo(cartComm_, globalDomainSize), dt);
        sim_ = std::make_unique<Simulation> (cartComm_, interComm_, getState(),
                                            checkpointInfo, gpuAwareMPI);
    }
    else
    {
        MPI_Check( MPI_Comm_dup(splitComm, &ioComm_) );
        MPI_Check( MPI_Intercomm_create(ioComm_,   localLeader, comm_, remoteLeader, tag, &interComm_) );

        MPI_Check( MPI_Comm_rank(ioComm_, &rank_) );

        post_ = std::make_unique<Postprocess> (ioComm_, interComm_, checkpointInfo.folder);
    }

    MPI_Check( MPI_Comm_free(&splitComm) );
}

void Mirheo::initLogger(MPI_Comm comm, LogInfo logInfo)
{
    if (logInfo.fileName == "stdout" ||
        logInfo.fileName == "stderr")
    {
        FileWrapper f(true);
        f.open(logInfo.fileName == "stdout" ?
               FileWrapper::SpecialStream::Cout :
               FileWrapper::SpecialStream::Cerr);
        logger.init(comm, std::move(f), logInfo.verbosityLvl);
    }
    else
    {
        logger.init(comm, logInfo.fileName+".log", logInfo.verbosityLvl);
    }
}

Mirheo::Mirheo(int3 nranks3D, real3 globalDomainSize, real dt,
               LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI)
{
    MPI_Init(nullptr, nullptr);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
    initializedMpi_ = true;

    init(nranks3D, globalDomainSize, dt, logInfo, checkpointInfo, gpuAwareMPI);
}

Mirheo::Mirheo(long commAddress, int3 nranks3D, real3 globalDomainSize, real dt,
               LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI)
{
    // see https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
    MPI_Comm comm = *((MPI_Comm*) commAddress);
    MPI_Comm_dup(comm, &comm_);
    init(nranks3D, globalDomainSize, dt, logInfo, checkpointInfo, gpuAwareMPI);    
}

Mirheo::Mirheo(MPI_Comm comm, int3 nranks3D, real3 globalDomainSize, real dt,
               LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI)
{
    MPI_Comm_dup(comm, &comm_);
    init(nranks3D, globalDomainSize, dt, logInfo, checkpointInfo, gpuAwareMPI);
}

static void safeCommFree(MPI_Comm *comm)
{
    if (*comm != MPI_COMM_NULL)
        MPI_Check( MPI_Comm_free(comm) );
}

Mirheo::~Mirheo()
{
    debug("Mirheo coordinator is destroyed");
    
    sim_.reset();
    post_.reset();

    safeCommFree(&comm_);
    safeCommFree(&cartComm_);
    safeCommFree(&ioComm_);
    safeCommFree(&compComm_);
    safeCommFree(&interComm_);
    
    if (initializedMpi_)
        MPI_Finalize();
}

void Mirheo::registerParticleVector(const std::shared_ptr<ParticleVector>& pv, const std::shared_ptr<InitialConditions>& ic)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->registerParticleVector(pv, ic);
}

void Mirheo::registerIntegrator(const std::shared_ptr<Integrator>& integrator)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->registerIntegrator(integrator);
}

void Mirheo::registerInteraction(const std::shared_ptr<Interaction>& interaction)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->registerInteraction(interaction);
}

void Mirheo::registerWall(const std::shared_ptr<Wall>& wall, int checkEvery)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->registerWall(wall, checkEvery);
}

void Mirheo::registerBouncer(const std::shared_ptr<Bouncer>& bouncer)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->registerBouncer(bouncer);
}

void Mirheo::registerObjectBelongingChecker (const std::shared_ptr<ObjectBelongingChecker>& checker, ObjectVector* ov)
{
    ensureNotInitialized();
    
    if (isComputeTask())
    {
        sim_->registerObjectBelongingChecker(checker);
        sim_->setObjectBelongingChecker(checker->getName(), ov->getName());
    }
}

void Mirheo::registerPlugins(const std::shared_ptr<SimulationPlugin>& simPlugin, const std::shared_ptr<PostprocessPlugin>& postPlugin)
{
    ensureNotInitialized();

    const int tag = pluginsTag_++;
    
    if (isComputeTask())
    {
        if ( simPlugin != nullptr && !(simPlugin->needPostproc() && noPostprocess_) )
            sim_->registerPlugin(simPlugin, tag);
    }
    else
    {
        if ( postPlugin != nullptr && !noPostprocess_ )
            post_->registerPlugin(postPlugin, tag);
    }
}

void Mirheo::registerPlugins(const PairPlugin &plugins) {
    registerPlugins(plugins.first, plugins.second);
}

void Mirheo::setIntegrator(Integrator *integrator, ParticleVector *pv)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->setIntegrator(integrator->getName(), pv->getName());
}

void Mirheo::setInteraction(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->setInteraction(interaction->getName(), pv1->getName(), pv2->getName());
}

void Mirheo::setBouncer(Bouncer *bouncer, ObjectVector *ov, ParticleVector *pv)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->setBouncer(bouncer->getName(), ov->getName(), pv->getName());
}

void Mirheo::setWallBounce(Wall *wall, ParticleVector *pv, real maximumPartTravel)
{
    ensureNotInitialized();
    
    if (isComputeTask())
        sim_->setWallBounce(wall->getName(), pv->getName(), maximumPartTravel);
}

MirState* Mirheo::getState()
{
    return state_.get();
}

const MirState* Mirheo::getState() const
{
    return state_.get();
}

Simulation* Mirheo::getSimulation()
{
    return sim_.get();
}

const Simulation* Mirheo::getSimulation() const
{
    return sim_.get();
}

std::shared_ptr<MirState> Mirheo::getMirState()
{
    return state_;
}

void Mirheo::dumpWalls2XDMF(std::vector<std::shared_ptr<Wall>> walls, real3 h, const std::string& filename)
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
        sim_->getWallByNameOrDie(wall->getName());
    }
    
    WallHelpers::dumpWalls2XDMF(sdfWalls, h, state_->domain, filename, sim_->getCartComm());
}

double Mirheo::computeVolumeInsideWalls(std::vector<std::shared_ptr<Wall>> walls, long nSamplesPerRank)
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
        sim_->getWallByNameOrDie(wall->getName());
    }

    return WallHelpers::volumeInsideWalls(sdfWalls, state_->domain, sim_->getCartComm(), nSamplesPerRank);
}

std::shared_ptr<ParticleVector> Mirheo::makeFrozenWallParticles(std::string pvName,
                                                               std::vector<std::shared_ptr<Wall>> walls,
                                                               std::vector<std::shared_ptr<Interaction>> interactions,
                                                               std::shared_ptr<Integrator> integrator,
                                                               real density, int nsteps)
{
    ensureNotInitialized();
    
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
        sim_->getWallByNameOrDie(wall->getName());

        info("Working with wall '%s'", wall->getCName());   
    }

    MirState stateCpy = *getState();
    
    Simulation wallsim(sim_->getCartComm(), MPI_COMM_NULL, getState());

    const real mass = 1.0_r;
    auto pv = std::make_shared<ParticleVector>(getState(), pvName, mass);
    auto ic = std::make_shared<UniformIC>(density);
    
    wallsim.registerParticleVector(pv, ic);
    
    wallsim.registerIntegrator(integrator);
    
    wallsim.setIntegrator (integrator->getName(),  pv->getName());

    for (auto& interaction : interactions) {
        wallsim.registerInteraction(interaction);        
        wallsim.setInteraction(interaction->getName(), pv->getName(), pv->getName());
    }
    
    wallsim.init();
    wallsim.run(nsteps);

    const real effectiveCutoff = wallsim.getMaxEffectiveCutoff();
    
    constexpr real wallThicknessTolerance = 0.2_r;
    constexpr real wallLevelSet = 0.0_r;
    const real wallThickness = effectiveCutoff + wallThicknessTolerance;

    info("wall thickness is set to %g", wallThickness);
    
    WallHelpers::freezeParticlesInWalls(sdfWalls, pv.get(), wallLevelSet, wallLevelSet + wallThickness);
    info("\n");

    sim_->registerParticleVector(pv, nullptr);

    for (auto &wall : walls)
        wall->attachFrozen(pv.get());

    // go back to initial state
    *state_ = stateCpy;
    
    return pv;
}

std::shared_ptr<ParticleVector> Mirheo::makeFrozenRigidParticles(std::shared_ptr<ObjectBelongingChecker> checker,
                                                                std::shared_ptr<ObjectVector> shape,
                                                                std::shared_ptr<InitialConditions> icShape,
                                                                std::vector<std::shared_ptr<Interaction>> interactions,
                                                                std::shared_ptr<Integrator>   integrator,
                                                                real density, int nsteps)
{
    ensureNotInitialized();
    
    if (!isComputeTask()) return nullptr;

    auto insideName = "inside_" + shape->getName();
    
    info("Generating frozen particles for rigid object '%s'...\n\n", shape->getCName());

    if (shape->local()->nObjects > 1)
        die("expected no more than one object vector; given %d", shape->local()->nObjects);
    

    const real mass = 1.0_r;
    auto pv = std::make_shared<ParticleVector>(getState(), "outside__" + shape->getName(), mass);
    auto ic = std::make_shared<UniformIC>(density);

    MirState stateCpy = *getState();

    {
        Simulation eqsim(sim_->getCartComm(), MPI_COMM_NULL, getState());
    
        eqsim.registerParticleVector(pv, ic);

        eqsim.registerIntegrator(integrator);
        eqsim.setIntegrator (integrator->getName(),  pv->getName());
        
        for (auto& interaction : interactions) {
            eqsim.registerInteraction(interaction);        
            eqsim.setInteraction(interaction->getName(), pv->getName(), pv->getName());
        }               
    
        eqsim.init();
        eqsim.run(nsteps);
    }

    Simulation freezesim(sim_->getCartComm(), MPI_COMM_NULL, getState());

    freezesim.registerParticleVector(pv, nullptr);
    freezesim.registerParticleVector(shape, icShape);
    freezesim.registerObjectBelongingChecker (checker);
    freezesim.setObjectBelongingChecker(checker->getName(), shape->getName());
    freezesim.applyObjectBelongingChecker(checker->getName(), pv->getName(), insideName, pv->getName(), 0);

    freezesim.init();
    freezesim.run(1);

    // go back to initial state
    *state_ = stateCpy;

    return freezesim.getSharedPVbyName(insideName);
}

std::shared_ptr<ParticleVector> Mirheo::applyObjectBelongingChecker(ObjectBelongingChecker *checker,
                                                                    ParticleVector *pv,
                                                                    int checkEvery,
                                                                    std::string inside,
                                                                    std::string outside)
{
    ensureNotInitialized();
    
    if (!isComputeTask()) return nullptr;
    
    if ( (inside != "" && outside != "") || (inside == "" && outside == "") )
        die("One and only one option can be specified for belonging checker '%s': inside or outside",
            checker->getCName());
    
    std::string newPVname;
    
    if (inside == "")
    {
        inside = pv->getName();
        newPVname = outside;
    }
    if (outside == "")
    {
        outside = pv->getName();
        newPVname = inside;
    }
        
    sim_->applyObjectBelongingChecker(checker->getName(), pv->getName(), inside, outside, checkEvery);
    return sim_->getSharedPVbyName(newPVname);
}

void Mirheo::sayHello()
{
    static const int max_length_version =  9;
    static const int max_length_sha1    = 46;
    std::string version = Version::mir_version;
    std::string sha1    = Version::git_SHA1;

    int missing_spaces = math::max(0, max_length_version - static_cast<int>(version.size()));
    version.append(missing_spaces, ' ');

    missing_spaces = math::max(0, max_length_sha1 - static_cast<int>(sha1.size()));
    sha1.append(missing_spaces, ' ');
    
    printf("\n");
    printf("**************************************************\n");
    printf("*                Mirheo %s                *\n", version.c_str());
    printf("* %s *\n", sha1.c_str());
    printf("**************************************************\n");
    printf("\n");
}

void Mirheo::setup()
{
    if (initialized_) return;
    
    if (isComputeTask())  sim_->init();
    else                 post_->init();
    
    initialized_ = true;
}

void Mirheo::ensureNotInitialized() const
{
    if (initialized_)
        die("Coordinator is already initialized.\n"
            "Do not call any register or set functions after 'restart' or 'run'");
}

void Mirheo::restart(std::string folder)
{
    folder = makePath(folder);

    setup();

    if (isComputeTask())  sim_->restart(folder);
    else                 post_->restart(folder);
}

bool Mirheo::isComputeTask() const
{
    return computeTask_ == 0;
}

bool Mirheo::isMasterTask() const
{
    return rank_ == 0 && isComputeTask();
}

bool Mirheo::isSimulationMasterTask() const
{
    return rank_ == 0 && isComputeTask();
}

bool Mirheo::isPostprocessMasterTask() const
{
    return rank_ == 0 && !isComputeTask();
}

void Mirheo::saveDependencyGraph_GraphML(std::string fname, bool current) const
{
    if (isComputeTask())
        sim_->saveDependencyGraph_GraphML(fname, current);
}

void Mirheo::startProfiler()
{
    if (isComputeTask())
        sim_->startProfiler();
}

void Mirheo::stopProfiler()
{
    if (isComputeTask())
        sim_->stopProfiler();
}

void Mirheo::run(int nsteps)
{
    setup();
    
    if (isComputeTask()) sim_->run(nsteps);
    else                post_->run();

    MPI_Check( MPI_Barrier(comm_) );
}


void Mirheo::logCompileOptions() const
{
    const std::string useNvtxOption = 
#ifdef USE_NVTX
        "ON"
#else
        "OFF"
#endif
        ;

    const std::string useDoubleOption = 
#ifdef MIRHEO_DOUBLE_PRECISION
        "ON"
#else
        "OFF"
#endif
        ;

    
    const std::string membraneDoubleOption = 
#ifdef MEMBRANE_FORCES_DOUBLE
        "ON"
#else
        "OFF"
#endif
        ;
    
    const std::string rodDoubleOption = 
#ifdef ROD_FORCES_DOUBLE
        "ON"
#else
        "OFF"
#endif
        ;
    
    
    info("compile time options:");
    info("USE_NVTX        : %s", useNvtxOption       .c_str());
    info("MIRHEO_DOUBLE   : %s", useDoubleOption     .c_str());
    info("MEMBRANE_DOUBLE : %s", membraneDoubleOption.c_str());
    info("ROD_DOUBLE      : %s", rodDoubleOption     .c_str());
}

static void storeToFile(const std::string& content, const std::string& filename)
{
    FileWrapper f;
    if (f.open(filename, "w") != FileWrapper::Status::Success)
        throw std::runtime_error("Error opening \"" + filename + "\", aborting.");
    fwrite(content.data(), 1, content.size(), f.get());
}

void Mirheo::writeSnapshot(std::string path)
{
    // Prepare context and the dumper.
    DumpContext context;
    context.path = path;
    context.groupComm = isComputeTask() ? compComm_ : ioComm_;
    Dumper dumper(std::move(context));

    // Create the snapshot folder before creating the dump.
    if (!path.empty())
        if (!createFoldersCollective(comm_, path))
            die("Error creating snapshot folder \"%s\", aborting.", path.c_str());

    // Dump the Mirheo object and the whole simulation recursively.
    Config::Dictionary dict;
    dict.emplace("__category", dumper("Mirheo"));
    dict.emplace("__type",     dumper("Mirheo"));
    if (state_)
        dict.emplace("state",  dumper(state_));
    if (sim_)
        dict.emplace("simulation", dumper(sim_));
    if (post_)
        dict.emplace("postprocess", dumper(post_));
    dumper.registerObject(this, std::move(dict));

    // Store the config files to the disk.
    if (isSimulationMasterTask() || isPostprocessMasterTask()) {
        std::string content = configToJSON(dumper.getConfig()) + '\n';
        std::string jsonName = isComputeTask() ? "config.compute.json" : "config.post.json";
        storeToFile(content, joinPaths(path, jsonName));
    }
}

} // namespace mirheo
