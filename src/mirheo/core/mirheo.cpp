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
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/compile_options.h>
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

LogInfo::LogInfo(const std::string& fileName_, int verbosityLvl_, bool noSplash_) :
    fileName(fileName_),
    verbosityLvl(verbosityLvl_),
    noSplash(noSplash_)
{}
MIRHEO_MEMBER_VARS(LogInfo, fileName, verbosityLvl, noSplash);

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
                  CheckpointInfo checkpointInfo, bool gpuAwareMPI,
                  LoaderContext *load)
{
    const ConfigValue *stateConfig = load ? &load->getConfig()["Mirheo"][0]["state"] : nullptr;

    int nranks;

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
        state_ = std::make_shared<MirState> (createDomainInfo(cartComm_, globalDomainSize),
                                             dt, stateConfig);
        sim_ = std::make_unique<Simulation> (cartComm_, MPI_COMM_NULL, getState(),
                                            checkpointInfo, gpuAwareMPI);
        computeTask_ = 0;
        return;
    }

    info("Program started, splitting communicator");

    MPI_Comm splitComm;
    
    // Note: Update `is*Task()` and `saveSnapshot` functions if modifying this.
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
        state_ = std::make_shared<MirState> (createDomainInfo(cartComm_, globalDomainSize),
                                             dt, stateConfig);
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

void Mirheo::initFromSnapshot(int3 nranks3D, const std::string &snapshotPath,
                           LogInfo logInfo, bool gpuAwareMPI)
{
    LoaderContext context{snapshotPath};
    Loader loader{&context};

    const ConfigValue& mirState = context.getConfig()["Mirheo"][0]["state"];
    real  dt               = mirState["dt"];
    real3 globalDomainSize = mirState["domainGlobalSize"];
    auto  checkpointInfo   = loader.load<CheckpointInfo>(
            context.getConfig()["Simulation"][0]["checkpointInfo"]);

    init(nranks3D, globalDomainSize, dt, logInfo, checkpointInfo, gpuAwareMPI, &context);
    loadSnapshot(this, loader);
}

void Mirheo::initLogger(MPI_Comm comm, LogInfo logInfo)
{
    if (logInfo.fileName == "stdout" ||
        logInfo.fileName == "stderr")
    {
        FileWrapper f;
        f.open(logInfo.fileName == "stdout" ?
               FileWrapper::SpecialStream::Cout :
               FileWrapper::SpecialStream::Cerr,
               true);
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

    initLogger(comm_, logInfo);
    init(nranks3D, globalDomainSize, dt, logInfo, checkpointInfo, gpuAwareMPI);
}

Mirheo::Mirheo(MPI_Comm comm, int3 nranks3D, real3 globalDomainSize, real dt,
               LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI)
{
    MPI_Comm_dup(comm, &comm_);
    initLogger(comm_, logInfo);
    init(nranks3D, globalDomainSize, dt, logInfo, checkpointInfo, gpuAwareMPI);
}

Mirheo::Mirheo(int3 nranks3D, const std::string &snapshotPath,
               LogInfo logInfo, bool gpuAwareMPI)
{
    MPI_Init(nullptr, nullptr);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
    initializedMpi_ = true;

    initLogger(comm_, logInfo);
    initFromSnapshot(nranks3D, snapshotPath, logInfo, gpuAwareMPI);
}

Mirheo::Mirheo(MPI_Comm comm, int3 nranks3D, const std::string &snapshotPath,
               LogInfo logInfo, bool gpuAwareMPI)
{
    MPI_Comm_dup(comm, &comm_);
    initLogger(comm_, logInfo);
    initFromSnapshot(nranks3D, snapshotPath, logInfo, gpuAwareMPI);
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

    std::vector<SDFBasedWall*> sdfWalls;
    for (auto &wall : walls)
    {
        auto sdfWall = dynamic_cast<SDFBasedWall*>(wall.get());
        if (sdfWall == nullptr)
            die("Only sdf-based walls are supported!");        
        else
            sdfWalls.push_back(sdfWall);

        // Check if the wall is set up
        sim_->getWallByNameOrDie(wall->getName());
    }
    
    wall_helpers::dumpWalls2XDMF(sdfWalls, h, state_->domain, filename, sim_->getCartComm());
}

double Mirheo::computeVolumeInsideWalls(std::vector<std::shared_ptr<Wall>> walls, long nSamplesPerRank)
{
    if (!isComputeTask()) return 0;

    info("Computing volume inside walls\n");
    
    std::vector<SDFBasedWall*> sdfWalls;
    for (auto &wall : walls)
    {
        auto sdfWall = dynamic_cast<SDFBasedWall*>(wall.get());
        if (sdfWall == nullptr)
            die("Only sdf-based walls are supported!");        
        else
            sdfWalls.push_back(sdfWall);

        // Check if the wall is set up
        sim_->getWallByNameOrDie(wall->getName());
    }

    return wall_helpers::volumeInsideWalls(sdfWalls, state_->domain, sim_->getCartComm(), nSamplesPerRank);
}

std::shared_ptr<ParticleVector> Mirheo::makeFrozenWallParticles(std::string pvName,
                                                               std::vector<std::shared_ptr<Wall>> walls,
                                                               std::vector<std::shared_ptr<Interaction>> interactions,
                                                               std::shared_ptr<Integrator> integrator,
                                                               real numDensity, real mass, int nsteps)
{
    ensureNotInitialized();
    
    if (!isComputeTask()) return nullptr;

    // Walls are not directly reusable in other simulations,
    // because they store some information like cell-lists
    //
    // But here we don't pass the wall into the other simulation,
    // we just use it to filter particles, which is totally fine
    
    info("Generating frozen particles for walls");

    std::vector<SDFBasedWall*> sdfWalls;

    for (auto &wall : walls)
    {
        if (auto sdfWall = dynamic_cast<SDFBasedWall*>(wall.get()))
            sdfWalls.push_back(sdfWall);
        else
            die("Only sdf-based walls are supported now! (%s is not)", wall->getCName());

        // Check if the wall is set up
        sim_->getWallByNameOrDie(wall->getName());

        info("Working with wall '%s'", wall->getCName());   
    }

    MirState stateCpy = *getState();
    
    Simulation wallsim(sim_->getCartComm(), MPI_COMM_NULL, getState(), CheckpointInfo{});

    auto pv = std::make_shared<ParticleVector>(getState(), pvName, mass);
    auto ic = std::make_shared<UniformIC>(numDensity);
    
    wallsim.registerParticleVector(pv, ic);
    
    wallsim.registerIntegrator(integrator);
    
    wallsim.setIntegrator (integrator->getName(),  pv->getName());

    for (auto& interaction : interactions)
    {
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
    
    wall_helpers::freezeParticlesInWalls(sdfWalls, pv.get(), wallLevelSet, wallLevelSet + wallThickness);
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
                                                                 real numDensity, real mass, int nsteps)
{
    ensureNotInitialized();
    
    if (!isComputeTask()) return nullptr;

    auto insideName = "inside_" + shape->getName();
    
    info("Generating frozen particles for rigid object '%s'...\n\n", shape->getCName());

    if (shape->local()->getNumObjects() > 1)
        die("expected no more than one object vector; given %d", shape->local()->getNumObjects());
    
    auto pv = std::make_shared<ParticleVector>(getState(), "outside__" + shape->getName(), mass);
    auto ic = std::make_shared<UniformIC>(numDensity);

    MirState stateCpy = *getState();

    {
        Simulation eqsim(sim_->getCartComm(), MPI_COMM_NULL, getState(), CheckpointInfo{});
    
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

    Simulation freezesim(sim_->getCartComm(), MPI_COMM_NULL, getState(), CheckpointInfo{});

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

void Mirheo::dumpDependencyGraphToGraphML(const std::string& fname, bool current) const
{
    if (isMasterTask())
        sim_->dumpDependencyGraphToGraphML(fname, current);
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
    info("compile time options:");
    info("MIRHEO_DOUBLE   : %d", compile_options.useDouble     );
    info("MEMBRANE_DOUBLE : %d", compile_options.membraneDouble);
    info("ROD_DOUBLE      : %d", compile_options.rodDouble     );
    info("USE_NVTX        : %d", compile_options.useNvtx       );
}

/// Prepare a ConfigObject listing all compilation options that affect the output format.
static ConfigObject compileOptionsToConfig(Saver& saver) {
    // Don't forget to update snapshot.cpp:checkCompilationOptions.
    ConfigObject out;
    out.unsafe_insert("useDouble", saver(compile_options.useDouble));
    return out;
}

void Mirheo::saveSnapshot(std::string path)
{
    // Abort if no-postprocess, not supported, mostly because loading assumes
    // both Simulation and Postprocess objects are always available.
    if (noPostprocess_)
        die("saveSnapshot not implemented for no-postprocess runs.");

    // Prepare context and the saver.
    SaverContext context;
    context.path = path;
    context.groupComm = isComputeTask() ? cartComm_ : ioComm_;
    if (context.groupComm == MPI_COMM_NULL)
        die("something's wrong with the comm.");
    Saver saver{&context};

    // Create the snapshot folder before saving.
    if (!path.empty())
        if (!createFoldersCollective(comm_, path))
            die("Error creating snapshot folder \"%s\", aborting.", path.c_str());

    // Dump the Mirheo object and the whole simulation recursively.
    ConfigValue::Object mir;
    mir.emplace("__category", saver("Mirheo"));
    mir.emplace("__type",     saver("Mirheo"));
    if (!attributes_.empty())
        mir.emplace("attributes", attributes_);
    if (state_)
        mir.emplace("state",  saver(state_));
    if (sim_)
        saver(sim_);
    if (post_)
        saver(post_);
    mir.emplace("compile_options", compileOptionsToConfig(saver));
    saver.registerObject(this, std::move(mir));
    ConfigObject localConfig = std::move(saver).getConfig();

    // Send the config to the postprocessing rank, merge and store on the disk.
    constexpr int simMaster = 0;   // With respect to comm_.
    constexpr int postMaster = 1;

    if (isSimulationMasterTask()) {
        std::string simJson = ConfigValue{std::move(localConfig)}.toJSONString();
        int size = (int)simJson.size();
        MPI_Check( MPI_Send(&size, 1, MPI_INT, postMaster, 98765, comm_) );
        MPI_Check( MPI_Send(simJson.c_str(), size, MPI_CHAR, postMaster, 98765, comm_) );
    }
    if (isPostprocessMasterTask()) {
        int size;
        MPI_Check( MPI_Recv(&size, 1, MPI_INT, simMaster, 98765, comm_, MPI_STATUS_IGNORE) );
        std::string simJson(size, '_');
        MPI_Check( MPI_Recv(const_cast<char *>(simJson.data()), size, MPI_CHAR,
                            simMaster, 98765, comm_, MPI_STATUS_IGNORE) );
        ConfigObject all = configFromJSON(simJson).getObject();

        // Merge post into sim (all).
        ConfigObject &post = localConfig;
        assert(post.size() == 2 || post.size() == 3);  // Plugins, postprocess, Mirheo.
        auto it = post.find("PostprocessPlugin");
        if (it != post.end()) {
            // Insert before `SimulationPlugin` for cosmetic reasons.
            all.unsafe_insert(all.find("SimulationPlugin"),
                              "PostprocessPlugin", std::move(it->second));
        }
        all.unsafe_insert(all.find("Simulation"),
                          "Postprocess", std::move(post["Postprocess"]));

        // Save the config.json.
        std::string content = ConfigValue(std::move(all)).toJSONString() + '\n';
        FileWrapper f(joinPaths(path, "config.json"), "w");
        fwrite(content.data(), 1, content.size(), f.get());
    }
    MPI_Barrier(comm_);
}

void Mirheo::setAttribute(const std::string& name, ConfigValue value)
{
    attributes_.insert_or_assign(name, std::move(value));
}
const ConfigValue& Mirheo::getAttribute(const std::string& name)
{
    return attributes_.at(name);
}

} // namespace mirheo
