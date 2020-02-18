#include "simulation.h"

#include <mirheo/core/bouncers/interface.h>
#include <mirheo/core/celllist.h>
#include <mirheo/core/exchangers/api.h>
#include <mirheo/core/initial_conditions/interface.h>
#include <mirheo/core/integrators/interface.h>
#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/managers/interactions.h>
#include <mirheo/core/mirheo_state.h>
#include <mirheo/core/object_belonging/interface.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/task_scheduler.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/restart_helpers.h>
#include <mirheo/core/walls/interface.h>

#include <algorithm>
#include <cuda_profiler_api.h>
#include <memory>
#include <set>

namespace mirheo
{

#define TASK_LIST(_)                                                    \
    _( checkpoint                          , "Checkpoint")              \
    _( cellLists                           , "Build cell-lists")        \
    _( integration                         , "Integration")             \
    _( partClearIntermediate               , "Particle clear intermediate") \
    _( partHaloIntermediateInit            , "Particle halo intermediate init") \
    _( partHaloIntermediateFinalize        , "Particle halo intermediate finalize") \
    _( localIntermediate                   , "Local intermediate")      \
    _( haloIntermediate                    , "Halo intermediate")       \
    _( accumulateInteractionIntermediate   , "Accumulate intermediate") \
    _( gatherInteractionIntermediate       , "Gather intermediate")     \
    _( partClearFinal                      , "Clear forces")            \
    _( partHaloFinalInit                   , "Particle halo final init") \
    _( partHaloFinalFinalize               , "Particle halo final finalize") \
    _( localForces                         , "Local forces")            \
    _( haloForces                          , "Halo forces")             \
    _( accumulateInteractionFinal          , "Accumulate forces")       \
    _( objHaloFinalInit                    , "Object halo final init")  \
    _( objHaloFinalFinalize                , "Object halo final finalize") \
    _( objHaloIntermediateInit             , "Object halo intermediate init")  \
    _( objHaloIntermediateFinalize         , "Object halo intermediate finalize") \
    _( objReverseIntermediateInit          , "Object reverse intermediate: init") \
    _( objReverseIntermediateFinalize      , "Object reverse intermediate: finalize") \
    _( objReverseFinalInit                 , "Object reverse final: init") \
    _( objReverseFinalFinalize             , "Object reverse final: finalize") \
    _( objClearLocalIntermediate           , "Clear local object intermediate") \
    _( objClearHaloIntermediate            , "Clear halo object intermediate") \
    _( objClearHaloForces                  , "Clear object halo forces") \
    _( objClearLocalForces                 , "Clear object local forces") \
    _( objLocalBounce                      , "Local object bounce")     \
    _( objHaloBounce                       , "Halo object bounce")      \
    _( correctObjBelonging                 , "Correct object belonging") \
    _( wallBounce                          , "Wall bounce")             \
    _( wallCheck                           , "Wall check")              \
    _( partRedistributeInit                , "Particle redistribute init") \
    _( partRedistributeFinalize            , "Particle redistribute finalize") \
    _( objRedistInit                       , "Object redistribute init") \
    _( objRedistFinalize                   , "Object redistribute finalize") \
    _( pluginsBeforeCellLists              , "Plugins: before cell lists") \
    _( pluginsBeforeForces                 , "Plugins: before forces")  \
    _( pluginsSerializeSend                , "Plugins: serialize and send") \
    _( pluginsBeforeIntegration            , "Plugins: before integration") \
    _( pluginsAfterIntegration             , "Plugins: after integration") \
    _( pluginsBeforeParticlesDistribution  , "Plugins: before particles distribution")


struct SimulationTasks
{
#define DECLARE(NAME, DESC) TaskScheduler::TaskID NAME ;

    TASK_LIST(DECLARE)

#undef DECLARE    
};

static void checkCartesianTopology(const MPI_Comm& cartComm)
{
    int topology;
    MPI_Check( MPI_Topo_test(cartComm, &topology) );

    if (topology != MPI_CART)
        die("Simulation expects a cartesian communicator");
}

struct Rank3DInfos
{
    int3 nranks3D, rank3D;
};

static Rank3DInfos getRank3DInfos(const MPI_Comm& cartComm)
{
    int nranks[3], periods[3], coords[3];
    checkCartesianTopology(cartComm);
    
    MPI_Check( MPI_Cart_get(cartComm, 3, nranks, periods, coords) );
    return {{nranks[0], nranks[1], nranks[2]},
            {coords[0], coords[1], coords[2]}};
}

static int getRank(const MPI_Comm& comm)
{
    int rank {0};
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    return rank;
}

Simulation::Simulation(const MPI_Comm &cartComm, const MPI_Comm &interComm, MirState *state,
                       CheckpointInfo checkpointInfo, bool gpuAwareMPI) :
    MirObject("simulation"),
    nranks3D_(getRank3DInfos(cartComm).nranks3D),
    rank3D_  (getRank3DInfos(cartComm).rank3D  ),
    cartComm_(cartComm),
    interComm_(interComm),
    state_(state),
    checkpointInfo_(checkpointInfo),
    rank_(getRank(cartComm)),
    scheduler_(std::make_unique<TaskScheduler>()),
    tasks_(std::make_unique<SimulationTasks>()),
    interactionsIntermediate_(std::make_unique<InteractionManager>()),
    interactionsFinal_(std::make_unique<InteractionManager>()),
    gpuAwareMPI_(gpuAwareMPI)
{
    if (checkpointInfo_.needDump())
        createFoldersCollective(cartComm_, checkpointInfo_.folder);

    const auto &domain = state_->domain;
    if (domain.globalSize.x <= 0 || domain.globalSize.y <= 0 || domain.globalSize.z <= 0) {
        die("Invalid domain size: [%f %f %f]",
            domain.globalSize.x, domain.globalSize.y, domain.globalSize.z);
    }
    info("Simulation initialized, subdomain size is [%f %f %f], subdomain starts "
         "at [%f %f %f]",
         domain.localSize.x, domain.localSize.y, domain.localSize.z,
         domain.globalStart.x, domain.globalStart.y, domain.globalStart.z);
}

Simulation::~Simulation() = default;

//================================================================================================
// Access for plugins
//================================================================================================

std::vector<ParticleVector*> Simulation::getParticleVectors() const
{
    std::vector<ParticleVector*> res;
    for (auto& pv : particleVectors_)
        res.push_back(pv.get());

    return res;
}

ParticleVector* Simulation::getPVbyName(const std::string& name) const
{
    auto pvIt = pvIdMap_.find(name);
    return (pvIt != pvIdMap_.end()) ? particleVectors_[pvIt->second].get() : nullptr;
}

std::shared_ptr<ParticleVector> Simulation::getSharedPVbyName(const std::string& name) const
{
    auto pvIt = pvIdMap_.find(name);
    return (pvIt != pvIdMap_.end()) ? particleVectors_[pvIt->second] : std::shared_ptr<ParticleVector>(nullptr);
}

ParticleVector* Simulation::getPVbyNameOrDie(const std::string& name) const
{
    auto pv = getPVbyName(name);
    if (pv == nullptr)
        die("No such particle vector: %s", getCName());
    return pv;
}

ObjectVector* Simulation::getOVbyNameOrDie(const std::string& name) const
{
    if (auto pv = getPVbyName(name))
    {
        if (auto ov = dynamic_cast<ObjectVector*>(pv))
            return ov;
        else
            die("'%s' is not an object vector", getCName());
    }
    else
    {
        die("No such object vector: %s", getCName());
    }
    return nullptr;
}

Wall* Simulation::getWallByNameOrDie(const std::string& name) const
{
    if (wallMap_.find(name) == wallMap_.end())
        die("No such wall: %s", getCName());

    auto it = wallMap_.find(name);
    return it->second.get();
}

CellList* Simulation::gelCellList(ParticleVector* pv) const
{
    auto clvecIt = cellListMap_.find(pv);
    if (clvecIt == cellListMap_.end())
        die("Particle Vector '%s' is not registered or broken", pv->getCName());

    if (clvecIt->second.size() == 0)
        return nullptr;
    else
        return clvecIt->second[0].get();
}

MPI_Comm Simulation::getCartComm() const
{
    return cartComm_;
}

int3 Simulation::getRank3D() const
{
    return rank3D_;
}

int3 Simulation::getNRanks3D() const
{
    return nranks3D_;
}

real Simulation::getCurrentDt() const
{
    return state_->dt;
}

real Simulation::getCurrentTime() const
{
    return static_cast<real>(state_->currentTime);
}

real Simulation::getMaxEffectiveCutoff() const
{
    const auto rcIntermediate = interactionsIntermediate_->getLargestCutoff();
    const auto rcFinal        = interactionsFinal_       ->getLargestCutoff();
    return rcIntermediate + rcFinal;
}

void Simulation::startProfiler() const
{
    CUDA_Check( cudaProfilerStart() );
}

void Simulation::stopProfiler() const
{
    CUDA_Check( cudaProfilerStop() );
}

//================================================================================================
// Registration
//================================================================================================

void Simulation::registerParticleVector(std::shared_ptr<ParticleVector> pv, std::shared_ptr<InitialConditions> ic)
{
    const std::string name = pv->getName();

    if (name == "none" || name == "all" || name == "")
        die("Invalid name for a particle vector (reserved word or empty): '%s'", getCName());
    
    if (pv->getName().rfind("_", 0) == 0)
        die("Identifier of Particle Vectors cannot start with _");

    if (pvIdMap_.find(name) != pvIdMap_.end())
        die("More than one particle vector is called %s", getCName());

    if (ic)
        ic->exec(cartComm_, pv.get(), 0);

    if (auto ov = dynamic_cast<ObjectVector*>(pv.get()))
    {
        info("Registered object vector '%s', %d objects, %d particles",
             getCName(), ov->local()->getNumObjects(), ov->local()->size());
        objectVectors_.push_back(ov);
    }
    else
    {
        info("Registered particle vector '%s', %d particles", getCName(), pv->local()->size());
    }

    particleVectors_.push_back(std::move(pv));
    pvIdMap_[name] = static_cast<int>(particleVectors_.size()) - 1;
}

void Simulation::registerWall(std::shared_ptr<Wall> wall, int every)
{
    const std::string name = wall->getName();

    if (wallMap_.find(name) != wallMap_.end())
        die("More than one wall is called %s", getCName());

    checkWallPrototypes_.push_back({wall.get(), every});

    // Let the wall know the particle vector associated with it
    wall->setup(cartComm_);

    info("Registered wall '%s'", getCName());

    wallMap_[name] = std::move(wall);
}

void Simulation::registerInteraction(std::shared_ptr<Interaction> interaction)
{
    const std::string name = interaction->getName();

    if (interactionMap_.find(name) != interactionMap_.end())
        die("More than one interaction is called %s", getCName());

    interactionMap_[name] = std::move(interaction);
}

void Simulation::registerIntegrator(std::shared_ptr<Integrator> integrator)
{
    const std::string name = integrator->getName();

    if (integratorMap_.find(name) != integratorMap_.end())
        die("More than one integrator is called %s", getCName());
    
    integratorMap_[name] = std::move(integrator);
}

void Simulation::registerBouncer(std::shared_ptr<Bouncer> bouncer)
{
    const std::string name = bouncer->getName();

    if (bouncerMap_.find(name) != bouncerMap_.end())
        die("More than one bouncer is called %s", getCName());

    bouncerMap_[name] = std::move(bouncer);
}

void Simulation::registerObjectBelongingChecker(std::shared_ptr<ObjectBelongingChecker> checker)
{
    const std::string name = checker->getName();

    if (belongingCheckerMap_.find(name) != belongingCheckerMap_.end())
        die("More than one splitter is called %s", getCName());

    belongingCheckerMap_[name] = std::move(checker);
}

void Simulation::registerPlugin(std::shared_ptr<SimulationPlugin> plugin, int tag)
{
    const std::string name = plugin->getName();

    bool found = false;
    for (auto& pl : plugins)
        if (pl->getName() == name) found = true;

    if (found)
        die("More than one plugin is called %s", getCName());

    plugin->setTag(tag);
    
    plugins.push_back(std::move(plugin));
}

//================================================================================================
// Applying something to something else
//================================================================================================

void Simulation::setIntegrator(const std::string& integratorName, const std::string& pvName)
{
    if (integratorMap_.find(integratorName) == integratorMap_.end())
        die("No such integrator: %s", integratorName.c_str());
    auto integrator = integratorMap_[integratorName].get();

    auto pv = getPVbyNameOrDie(pvName);

    if (pvsIntegratorMap_.find(pvName) != pvsIntegratorMap_.end())
        die("particle vector '%s' already set to integrator '%s'",
            pvName.c_str(), pvsIntegratorMap_[pvName].c_str());

    pvsIntegratorMap_[pvName] = integratorName;
    
    integrator->setPrerequisites(pv);

    integratorPrototypes_.push_back({pv, integrator});
}

void Simulation::setInteraction(const std::string& interactionName, const std::string& pv1Name, const std::string& pv2Name)
{
    auto pv1 = getPVbyNameOrDie(pv1Name);
    auto pv2 = getPVbyNameOrDie(pv2Name);

    if (interactionMap_.find(interactionName) == interactionMap_.end())
        die("No such interaction: %s", interactionName.c_str());
    auto interaction = interactionMap_[interactionName].get();    

    const real rc = interaction->getCutoffRadius();
    interactionPrototypes_.push_back({rc, pv1, pv2, interaction});
}

void Simulation::setBouncer(const std::string& bouncerName, const std::string& objName, const std::string& pvName)
{
    auto pv = getPVbyNameOrDie(pvName);
    auto ov = getOVbyNameOrDie(objName);

    if (bouncerMap_.find(bouncerName) == bouncerMap_.end())
        die("No such bouncer: %s", bouncerName.c_str());
    auto bouncer = bouncerMap_[bouncerName].get();

    bouncer->setup(ov);
    bouncer->setPrerequisites(pv);
    bouncerPrototypes_.push_back({bouncer, pv});
}

void Simulation::setWallBounce(const std::string& wallName, const std::string& pvName, real maximumPartTravel)
{
    auto pv = getPVbyNameOrDie(pvName);

    if (wallMap_.find(wallName) == wallMap_.end())
        die("No such wall: %s", wallName.c_str());
    auto wall = wallMap_[wallName].get();

    if (auto ov = dynamic_cast<ObjectVector*>(pv))
        die("Object Vectors can not be bounced from walls in the current implementaion. "
            "Invalid combination: wall '%s' and OV '%s'", wall->getCName(), ov->getCName());

    wall->setPrerequisites(pv);
    wallPrototypes_.push_back( {wall, pv, maximumPartTravel} );
}

void Simulation::setObjectBelongingChecker(const std::string& checkerName, const std::string& objName)
{
    if (belongingCheckerMap_.find(checkerName) == belongingCheckerMap_.end())
        die("No such belonging checker: %s", checkerName.c_str());
    auto checker = belongingCheckerMap_[checkerName].get();

    if (auto ov = dynamic_cast<ObjectVector*>(getPVbyNameOrDie(objName)))
        checker->setup(ov);
    else
        die("'%s' is not an object vector: can not set to belonging '%s'",
            objName.c_str(), checkerName.c_str());        
}


void Simulation::applyObjectBelongingChecker(const std::string& checkerName, const std::string& source,
                                             const std::string& inside, const std::string& outside,
                                             int checkEvery)
{
    auto pvSource = getPVbyNameOrDie(source);

    if (inside == outside)
        die("Splitting into same pvs: %s into %s %s",
            source.c_str(), inside.c_str(), outside.c_str());

    if (source != inside && source != outside)
        die("At least one of the split destinations should be the same as source: %s into %s %s",
            source.c_str(), inside.c_str(), outside.c_str());

    if (belongingCheckerMap_.find(checkerName) == belongingCheckerMap_.end())
        die("No such belonging checker: %s", checkerName.c_str());

    if (getPVbyName(inside) != nullptr && inside != source)
        die("Cannot split into existing particle vector: %s into %s %s",
            source.c_str(), inside.c_str(), outside.c_str());

    if (getPVbyName(outside) != nullptr && outside != source)
        die("Cannot split into existing particle vector: %s into %s %s",
            source.c_str(), inside.c_str(), outside.c_str());


    auto checker = belongingCheckerMap_[checkerName].get();

    std::shared_ptr<ParticleVector> pvInside, pvOutside;
    std::shared_ptr<InitialConditions> noIC;

    if (inside != "none" && getPVbyName(inside) == nullptr)
    {
        pvInside = std::make_shared<ParticleVector> (state_, inside, pvSource->getMassPerParticle());
        registerParticleVector(pvInside, noIC);
    }

    if (outside != "none" && getPVbyName(outside) == nullptr)
    {
        pvOutside = std::make_shared<ParticleVector> (state_, outside, pvSource->getMassPerParticle());
        registerParticleVector(pvOutside, noIC);
    }

    splitterPrototypes_.push_back({checker, pvSource, getPVbyName(inside), getPVbyName(outside)});

    belongingCorrectionPrototypes_.push_back({checker, getPVbyName(inside), getPVbyName(outside), checkEvery});
}

static void sortDescendingOrder(std::vector<real>& v)
{
    std::sort(v.begin(), v.end(), [] (real a, real b) { return a > b; });
}

// assume sorted array (ascending or descending)
static void removeDuplicatedElements(std::vector<real>& v, real tolerance)
{
    auto it = std::unique(v.begin(), v.end(), [=] (real a, real b) { return math::abs(a - b) < tolerance; });
    v.resize( std::distance(v.begin(), it) );    
}

void Simulation::prepareCellLists()
{
    info("Preparing cell-lists");

    std::map<ParticleVector*, std::vector<real>> cutOffMap;

    // Deal with the cell-lists and interactions
    for (auto prototype : interactionPrototypes_)
    {
        real rc = prototype.rc;
        cutOffMap[prototype.pv1].push_back(rc);
        cutOffMap[prototype.pv2].push_back(rc);
    }

    for (auto& cutoffPair : cutOffMap)
    {
        auto& pv      = cutoffPair.first;
        auto& cutoffs = cutoffPair.second;

        sortDescendingOrder(cutoffs);
        removeDuplicatedElements(cutoffs, rcTolerance_);

        bool primary = true;

        // Don't use primary cell-lists with ObjectVectors
        if (dynamic_cast<ObjectVector*>(pv))
            primary = false;

        for (auto rc : cutoffs)
        {
            cellListMap_[pv].push_back(primary ?
                    std::make_unique<PrimaryCellList>(pv, rc, state_->domain.localSize) :
                    std::make_unique<CellList>       (pv, rc, state_->domain.localSize));
            primary = false;
        }
    }

    for (auto& pv : particleVectors_)
    {
        auto pvptr = pv.get();
        if (cellListMap_[pvptr].empty())
        {
            const real defaultRc = 1._r;
            bool primary = true;

            // Don't use primary cell-lists with ObjectVectors
            if (dynamic_cast<ObjectVector*>(pvptr))
                primary = false;

            cellListMap_[pvptr].push_back
                (primary ?
                 std::make_unique<PrimaryCellList>(pvptr, defaultRc, state_->domain.localSize) :
                 std::make_unique<CellList>       (pvptr, defaultRc, state_->domain.localSize));
        }
    }
}

// Choose a CL with smallest but bigger than rc cell
static CellList* selectBestClist(std::vector<std::unique_ptr<CellList>>& cellLists, real rc, real tolerance)
{
    real minDiff = 1e6;
    CellList* best = nullptr;
    
    for (auto& cl : cellLists) {
        real diff = cl->rc - rc;
        if (diff > -tolerance && diff < minDiff) {
            best    = cl.get();
            minDiff = diff;
        }
    }
    return best;
}

void Simulation::prepareInteractions()
{
    info("Preparing interactions");

    for (auto& prototype : interactionPrototypes_)
    {
        auto  rc = prototype.rc;
        auto pv1 = prototype.pv1;
        auto pv2 = prototype.pv2;

        auto& clVec1 = cellListMap_[pv1];
        auto& clVec2 = cellListMap_[pv2];

        CellList *cl1, *cl2;

        cl1 = selectBestClist(clVec1, rc, rcTolerance_);
        cl2 = selectBestClist(clVec2, rc, rcTolerance_);
        
        auto inter = prototype.interaction;

        inter->setPrerequisites(pv1, pv2, cl1, cl2);

        if (inter->getStage() == Interaction::Stage::Intermediate)
            interactionsIntermediate_->add(inter, pv1, pv2, cl1, cl2);
        else
            interactionsFinal_       ->add(inter, pv1, pv2, cl1, cl2);
    }
}

void Simulation::prepareBouncers()
{
    info("Preparing object bouncers");

    for (auto& prototype : bouncerPrototypes_)
    {
        auto bouncer = prototype.bouncer;
        auto pv      = prototype.pv;

        if (pvsIntegratorMap_.find(pv->getName()) == pvsIntegratorMap_.end())
            die("Setting bouncer '%s': particle vector '%s' has no integrator, required for bounce back",
                bouncer->getCName(), pv->getCName());
        
        auto& clVec = cellListMap_[pv];

        if (clVec.empty()) continue;

        CellList *cl = clVec[0].get();

        regularBouncers_.push_back([bouncer, pv, cl] (cudaStream_t stream) {
            bouncer->bounceLocal(pv, cl, stream);
        });

        haloBouncers_.   push_back([bouncer, pv, cl] (cudaStream_t stream) {
            bouncer->bounceHalo (pv, cl, stream);
        });
    }
}

void Simulation::prepareWalls()
{
    info("Preparing walls");

    for (auto& prototype : wallPrototypes_)
    {
        auto wall = prototype.wall;
        auto pv   = prototype.pv;
        
        auto& clVec = cellListMap_[pv];

        if (clVec.empty()) continue;

        CellList *cl = clVec[0].get();

        wall->attach(pv, cl, prototype.maximumPartTravel);
    }

    for (auto& wall : wallMap_)
    {
        auto wallPtr = wall.second.get();

        // All the particles should be removed from within the wall,
        // even those that do not interact with it
        // Only frozen wall particles will remain
        for (auto& anypv : particleVectors_)
            wallPtr->removeInner(anypv.get());
    }
}

void Simulation::preparePlugins()
{
    info("Preparing plugins");
    for (auto& pl : plugins) {
        debug("Setup and handshake of plugin %s", pl->getCName());
        pl->setup(this, cartComm_, interComm_);
        pl->handshake();
    }
    info("done Preparing plugins");
}


std::vector<std::string> Simulation::getExtraDataToExchange(ObjectVector *ov)
{
    std::set<std::string> channels;
    
    for (auto& entry : bouncerMap_)
    {
        auto& bouncer = entry.second;
        if (bouncer->getObjectVector() != ov) continue;

        auto extraChannels = bouncer->getChannelsToBeExchanged();
        for (auto channel : extraChannels)
            channels.insert(channel);
    }

    for (auto& entry : belongingCheckerMap_)
    {
        auto& belongingChecker = entry.second;
        if (belongingChecker->getObjectVector() != ov) continue;

        auto extraChannels = belongingChecker->getChannelsToBeExchanged();
        for (auto channel : extraChannels)
            channels.insert(channel);
    }

    return {channels.begin(), channels.end()};
}

std::vector<std::string> Simulation::getDataToSendBack(const std::vector<std::string>& extraOut,
                                                       ObjectVector *ov)
{
    std::set<std::string> channels;

    for (const auto& name : extraOut)
        channels.insert(name);
    
    for (auto& entry : bouncerMap_)
    {
        auto& bouncer = entry.second;
        if (bouncer->getObjectVector() != ov) continue;

        auto extraChannels = bouncer->getChannelsToBeSentBack();
        for (auto channel : extraChannels)
            channels.insert(channel);
    }

    return {channels.begin(), channels.end()};
}

void Simulation::prepareEngines()
{
    auto partRedistImp                  = std::make_unique<ParticleRedistributor>();
    auto partHaloFinalImp               = std::make_unique<ParticleHaloExchanger>();
    auto partHaloIntermediateImp        = std::make_unique<ParticleHaloExchanger>();
    auto objRedistImp                   = std::make_unique<ObjectRedistributor>();        
    auto objHaloFinalImp                = std::make_unique<ObjectHaloExchanger>();
    auto objHaloIntermediateImp         = std::make_unique<ObjectExtraExchanger>  (objHaloFinalImp.get());
    auto objHaloReverseIntermediateImp  = std::make_unique<ObjectReverseExchanger>(objHaloFinalImp.get());
    auto objHaloReverseFinalImp         = std::make_unique<ObjectReverseExchanger>(objHaloFinalImp.get());

    debug("Attaching particle vectors to halo exchanger and redistributor");
    for (auto& pv : particleVectors_)
    {
        auto  pvPtr       = pv.get();
        auto& cellListVec = cellListMap_[pvPtr];        

        if (cellListVec.size() == 0) continue;

        CellList *clInt = interactionsIntermediate_ ->getLargestCellList(pvPtr);
        CellList *clOut = interactionsFinal_        ->getLargestCellList(pvPtr);

        auto extraInt = interactionsIntermediate_->getOutputChannels(pvPtr);
        auto extraOut = interactionsFinal_       ->getOutputChannels(pvPtr);

        auto cl = cellListVec[0].get();
        
        if (auto ov = dynamic_cast<ObjectVector*>(pvPtr))
        {
            objRedistImp->attach(ov);

            auto extraToExchange = getExtraDataToExchange(ov);
            auto reverseExchange = getDataToSendBack(extraInt, ov);

            objHaloFinalImp->attach(ov, cl->rc, extraToExchange); // always active because of bounce back; TODO: check if bounce back is active
            objHaloReverseFinalImp->attach(ov, extraOut);

            objHaloIntermediateImp->attach(ov, extraInt);
            objHaloReverseIntermediateImp->attach(ov, reverseExchange);
        }
        else
        {
            partRedistImp->attach(pvPtr, cl);
            
            if (clInt != nullptr)
                partHaloIntermediateImp->attach(pvPtr, clInt, {});

            if (clOut != nullptr)
                partHaloFinalImp->attach(pvPtr, clOut, extraInt);
        }
    }
    
    std::function< std::unique_ptr<ExchangeEngine>(std::unique_ptr<Exchanger>) > makeEngine;
    
    // If we're on one node, use a singleNode engine
    // otherwise use MPI
    if (nranks3D_.x * nranks3D_.y * nranks3D_.z == 1)
        makeEngine = [this] (std::unique_ptr<Exchanger> exch) {
            return std::make_unique<SingleNodeEngine> (std::move(exch));
        };
    else
        makeEngine = [this] (std::unique_ptr<Exchanger> exch) {
            return std::make_unique<MPIExchangeEngine> (std::move(exch), cartComm_, gpuAwareMPI_);
        };
    
    partRedistributor_            = makeEngine(std::move(partRedistImp));
    partHaloFinal_                = makeEngine(std::move(partHaloFinalImp));
    partHaloIntermediate_         = makeEngine(std::move(partHaloIntermediateImp));
    objRedistibutor_              = makeEngine(std::move(objRedistImp));
    objHaloFinal_                 = makeEngine(std::move(objHaloFinalImp));
    objHaloIntermediate_          = makeEngine(std::move(objHaloIntermediateImp));
    objHaloReverseIntermediate_   = makeEngine(std::move(objHaloReverseIntermediateImp));
    objHaloReverseFinal_          = makeEngine(std::move(objHaloReverseFinalImp));
}

void Simulation::execSplitters()
{
    info("Splitting particle vectors with respect to object belonging");

    for (auto& prototype : splitterPrototypes_)
    {
        auto checker = prototype.checker;
        auto src     = prototype.pvSrc;
        auto inside  = prototype.pvIn;
        auto outside = prototype.pvOut;

        checker->splitByBelonging(src, inside, outside, 0);
    }
}

void Simulation::createTasks()
{
#define INIT(NAME, DESC) tasks_ -> NAME = scheduler_->createTask(DESC);
    TASK_LIST(INIT);
#undef INIT

    if (checkpointInfo_.every > 0)
        scheduler_->addTask(tasks_->checkpoint,
                           [this](__UNUSED cudaStream_t stream) { this->checkpoint(); },
                           checkpointInfo_.every);

    for (auto& clVec : cellListMap_)
        for (auto& cl : clVec.second)
        {
            auto clPtr = cl.get();
            scheduler_->addTask(tasks_->cellLists, [clPtr] (cudaStream_t stream) { clPtr->build(stream); } );
        }

    // Only particle forces, not object ones here
    for (auto& pv : particleVectors_)
    {
        auto pvPtr = pv.get();
        scheduler_->addTask(tasks_->partClearIntermediate,
                           [this, pvPtr] (cudaStream_t stream)
        {
            interactionsIntermediate_->clearOutput(pvPtr, stream);
            interactionsFinal_       ->clearInput (pvPtr, stream);
        } );

        scheduler_->addTask(tasks_->partClearFinal,
                           [this, pvPtr] (cudaStream_t stream) { interactionsFinal_->clearOutput(pvPtr, stream); } );
    }

    for (auto& pl : plugins)
    {
        auto plPtr = pl.get();

        scheduler_->addTask(tasks_->pluginsBeforeCellLists, [plPtr, this] (cudaStream_t stream) {
            plPtr->beforeCellLists(stream);
        });

        scheduler_->addTask(tasks_->pluginsBeforeForces, [plPtr, this] (cudaStream_t stream) {
            plPtr->beforeForces(stream);
        });

        scheduler_->addTask(tasks_->pluginsSerializeSend, [plPtr] (cudaStream_t stream) {
            plPtr->serializeAndSend(stream);
        });

        scheduler_->addTask(tasks_->pluginsBeforeIntegration, [plPtr] (cudaStream_t stream) {
            plPtr->beforeIntegration(stream);
        });

        scheduler_->addTask(tasks_->pluginsAfterIntegration, [plPtr] (cudaStream_t stream) {
            plPtr->afterIntegration(stream);
        });

        scheduler_->addTask(tasks_->pluginsBeforeParticlesDistribution, [plPtr] (cudaStream_t stream) {
            plPtr->beforeParticleDistribution(stream);
        });
    }


    // If we have any non-object vectors
    if (particleVectors_.size() != objectVectors_.size())
    {
        scheduler_->addTask(tasks_->partHaloIntermediateInit, [this] (cudaStream_t stream) {
            partHaloIntermediate_->init(stream);
        });

        scheduler_->addTask(tasks_->partHaloIntermediateFinalize, [this] (cudaStream_t stream) {
            partHaloIntermediate_->finalize(stream);
        });

        scheduler_->addTask(tasks_->partHaloFinalInit, [this] (cudaStream_t stream) {
            partHaloFinal_->init(stream);
        });

        scheduler_->addTask(tasks_->partHaloFinalFinalize, [this] (cudaStream_t stream) {
            partHaloFinal_->finalize(stream);
        });

        scheduler_->addTask(tasks_->partRedistributeInit, [this] (cudaStream_t stream) {
            partRedistributor_->init(stream);
        });

        scheduler_->addTask(tasks_->partRedistributeFinalize, [this] (cudaStream_t stream) {
            partRedistributor_->finalize(stream);
        });
    }


    scheduler_->addTask(tasks_->localIntermediate,
                       [this] (cudaStream_t stream) {
                           interactionsIntermediate_->executeLocal(stream);
                       });

    scheduler_->addTask(tasks_->haloIntermediate,
                       [this] (cudaStream_t stream) {
                           interactionsIntermediate_->executeHalo(stream);
                       });

    scheduler_->addTask(tasks_->localForces,
                       [this] (cudaStream_t stream) {
                           interactionsFinal_->executeLocal(stream);
                       });

    scheduler_->addTask(tasks_->haloForces,
                       [this] (cudaStream_t stream) {
                           interactionsFinal_->executeHalo(stream);
                       });
    

    scheduler_->addTask(tasks_->gatherInteractionIntermediate,
                       [this] (cudaStream_t stream) {
                           interactionsFinal_->gatherInputToCells(stream);
                       });

    scheduler_->addTask(tasks_->accumulateInteractionIntermediate,
                       [this] (cudaStream_t stream) {
                           interactionsIntermediate_->accumulateOutput(stream);
                       });
            
    scheduler_->addTask(tasks_->accumulateInteractionFinal,
                       [this] (cudaStream_t stream) {
                           interactionsFinal_->accumulateOutput(stream);
                       });


    for (const auto& prototype : integratorPrototypes_)
    {
        auto pv         = prototype.pv;
        auto integrator = prototype.integrator;
        scheduler_->addTask(tasks_->integration, [integrator, pv] (cudaStream_t stream)
        {
            integrator->execute(pv, stream);
        });
    }


    // As there are no primary cell-lists for objects
    // we need to separately clear real obj forces and forces in the cell-lists
    for (auto ov : objectVectors_)
    {
        scheduler_->addTask(tasks_->objClearLocalIntermediate, [this, ov] (cudaStream_t stream)
        {
            interactionsIntermediate_->clearOutput(ov, stream);
            interactionsIntermediate_->clearOutputLocalPV(ov, ov->local(), stream);

            interactionsFinal_->clearInput(ov, stream);
            interactionsFinal_->clearInputLocalPV(ov, ov->local(), stream);
        });

        scheduler_->addTask(tasks_->objClearHaloIntermediate, [this, ov] (cudaStream_t stream)
        {
            interactionsIntermediate_->clearOutputLocalPV(ov, ov->halo(), stream);
            interactionsFinal_       ->clearInputLocalPV(ov, ov->halo(), stream);
        });

        scheduler_->addTask(tasks_->objClearLocalForces, [this, ov] (cudaStream_t stream)
        {
            auto lov = ov->local();
            interactionsFinal_->clearOutputLocalPV(ov, lov, stream);
            interactionsFinal_->clearOutput(ov, stream);
            lov->getMeshForces(stream)->clear(stream);

            // force clear forces in case there is no interactions but bounce back
            if (interactionsFinal_->empty())
                lov->forces().clearDevice(stream);

            if (auto rov = dynamic_cast<RigidObjectVector*>(ov))
                rov->local()->clearRigidForces(stream);
        });

        scheduler_->addTask(tasks_->objClearHaloForces, [this, ov] (cudaStream_t stream)
        {
            auto lov = ov->halo();
            interactionsFinal_->clearOutputLocalPV(ov, lov, stream);
            lov->getMeshForces(stream)->clear(stream);

            // force clear forces in case there is no interactions but bounce back
            if (interactionsFinal_->empty())
                lov->forces().clearDevice(stream);

            if (auto rov = dynamic_cast<RigidObjectVector*>(ov))
                rov->halo()->clearRigidForces(stream);
        });
    }

    for (auto& bouncer : regularBouncers_)
        scheduler_->addTask(tasks_->objLocalBounce, [bouncer, this] (cudaStream_t stream)
        {
            bouncer(stream);
        });

    for (auto& bouncer : haloBouncers_)
        scheduler_->addTask(tasks_->objHaloBounce, [bouncer, this] (cudaStream_t stream)
        {
            bouncer(stream);
        });

    for (auto& prototype : belongingCorrectionPrototypes_)
    {
        auto checker = prototype.checker;
        auto pvIn    = prototype.pvIn;
        auto pvOut   = prototype.pvOut;
        auto every   = prototype.every;

        if (every > 0)
        {
            scheduler_->addTask(tasks_->correctObjBelonging, [checker, pvIn, pvOut] (cudaStream_t stream) {
                if (pvIn  != nullptr) checker->splitByBelonging(pvIn,  pvIn, pvOut, stream);
                if (pvOut != nullptr) checker->splitByBelonging(pvOut, pvIn, pvOut, stream);
            }, every);
        }
    }

    if (objectVectors_.size() > 0)
    {
        scheduler_->addTask(tasks_->objHaloIntermediateInit, [this] (cudaStream_t stream) {
            objHaloIntermediate_->init(stream);
        });

        scheduler_->addTask(tasks_->objHaloIntermediateFinalize, [this] (cudaStream_t stream) {
            objHaloIntermediate_->finalize(stream);
        });

        scheduler_->addTask(tasks_->objHaloFinalInit, [this] (cudaStream_t stream) {
            objHaloFinal_->init(stream);
        });

        scheduler_->addTask(tasks_->objHaloFinalFinalize, [this] (cudaStream_t stream) {
            objHaloFinal_->finalize(stream);
        });

        scheduler_->addTask(tasks_->objReverseIntermediateInit, [this] (cudaStream_t stream) {
            objHaloReverseIntermediate_->init(stream);
        });

        scheduler_->addTask(tasks_->objReverseIntermediateFinalize, [this] (cudaStream_t stream) {
            objHaloReverseIntermediate_->finalize(stream);
        });

        scheduler_->addTask(tasks_->objReverseFinalInit, [this] (cudaStream_t stream) {
            objHaloReverseFinal_->init(stream);
        });

        scheduler_->addTask(tasks_->objReverseFinalFinalize, [this] (cudaStream_t stream) {
            objHaloReverseFinal_->finalize(stream);
        });

        scheduler_->addTask(tasks_->objRedistInit, [this] (cudaStream_t stream) {
            objRedistibutor_->init(stream);
        });

        scheduler_->addTask(tasks_->objRedistFinalize, [this] (cudaStream_t stream) {
            objRedistibutor_->finalize(stream);
        });
    }

    for (auto& wall : wallMap_)
    {
        auto wallPtr = wall.second.get();
        scheduler_->addTask(tasks_->wallBounce, [wallPtr, this] (cudaStream_t stream) {    
            wallPtr->bounce(stream);
        });
    }

    for (auto& prototype : checkWallPrototypes_)
    {
        auto wall  = prototype.wall;
        auto every = prototype.every;

        if (every > 0)
            scheduler_->addTask(tasks_->wallCheck, [this, wall] (cudaStream_t stream) { wall->check(stream); }, every);
    }
}

static void createTasksDummy(TaskScheduler *scheduler, SimulationTasks *tasks)
{
#define INIT(NAME, DESC) tasks -> NAME = scheduler->createTask(DESC);
#define DUMMY_TASK(NAME, DESC) scheduler->addTask(tasks->NAME, [](cudaStream_t) {info("executing " DESC);});

    TASK_LIST(INIT);
    TASK_LIST(DUMMY_TASK);

#undef INIT
#undef DUMMY_TASK
}

static void buildDependencies(TaskScheduler *scheduler, SimulationTasks *tasks)
{
    scheduler->addDependency(tasks->pluginsBeforeCellLists, { tasks->cellLists }, {});
    
    scheduler->addDependency(tasks->checkpoint, { tasks->partClearFinal }, { tasks->cellLists });

    scheduler->addDependency(tasks->correctObjBelonging, { tasks->cellLists }, {});

    scheduler->addDependency(tasks->cellLists, {tasks->partClearFinal, tasks->partClearIntermediate, tasks->objClearLocalIntermediate}, {});

    
    scheduler->addDependency(tasks->pluginsBeforeForces, {tasks->localForces, tasks->haloForces}, {tasks->partClearFinal});
    scheduler->addDependency(tasks->pluginsSerializeSend, {tasks->pluginsBeforeIntegration, tasks->pluginsAfterIntegration}, {tasks->pluginsBeforeForces});

    scheduler->addDependency(tasks->objReverseFinalInit, {}, {tasks->haloForces});
    scheduler->addDependency(tasks->objReverseFinalFinalize, {tasks->accumulateInteractionFinal}, {tasks->objReverseFinalInit});

    scheduler->addDependency(tasks->localIntermediate, {}, {tasks->partClearIntermediate, tasks->objClearLocalIntermediate});
    scheduler->addDependency(tasks->partHaloIntermediateInit, {}, {tasks->partClearIntermediate, tasks->cellLists});
    scheduler->addDependency(tasks->partHaloIntermediateFinalize, {}, {tasks->partHaloIntermediateInit});

    scheduler->addDependency(tasks->objClearHaloIntermediate, {}, {tasks->cellLists});
    scheduler->addDependency(tasks->haloIntermediate, {}, {tasks->partHaloIntermediateFinalize, tasks->objClearHaloIntermediate});
    scheduler->addDependency(tasks->objReverseIntermediateInit, {}, {tasks->haloIntermediate});    
    scheduler->addDependency(tasks->objReverseIntermediateFinalize, {}, {tasks->objReverseIntermediateInit});

    scheduler->addDependency(tasks->accumulateInteractionIntermediate, {}, {tasks->localIntermediate, tasks->haloIntermediate});
    scheduler->addDependency(tasks->gatherInteractionIntermediate, {}, {tasks->accumulateInteractionIntermediate, tasks->objReverseIntermediateFinalize});

    scheduler->addDependency(tasks->localForces, {}, {tasks->gatherInteractionIntermediate});

    scheduler->addDependency(tasks->objHaloIntermediateInit, {}, {tasks->gatherInteractionIntermediate});
    scheduler->addDependency(tasks->objHaloIntermediateFinalize, {}, {tasks->objHaloIntermediateInit});
    
    scheduler->addDependency(tasks->partHaloFinalInit, {}, {tasks->pluginsBeforeForces, tasks->gatherInteractionIntermediate, tasks->objHaloIntermediateInit});
    scheduler->addDependency(tasks->partHaloFinalFinalize, {}, {tasks->partHaloFinalInit});

    scheduler->addDependency(tasks->haloForces, {}, {tasks->partHaloFinalFinalize, tasks->objHaloIntermediateFinalize});
    scheduler->addDependency(tasks->accumulateInteractionFinal, {tasks->integration}, {tasks->haloForces, tasks->localForces});

    scheduler->addDependency(tasks->pluginsBeforeIntegration, {tasks->integration}, {tasks->accumulateInteractionFinal});
    scheduler->addDependency(tasks->wallBounce, {}, {tasks->integration});
    scheduler->addDependency(tasks->wallCheck, {tasks->partRedistributeInit}, {tasks->wallBounce});

    scheduler->addDependency(tasks->objHaloFinalInit, {}, {tasks->integration, tasks->objRedistFinalize});
    scheduler->addDependency(tasks->objHaloFinalFinalize, {}, {tasks->objHaloFinalInit});

    scheduler->addDependency(tasks->objClearHaloForces, {tasks->objHaloBounce}, {tasks->objHaloFinalFinalize});
    scheduler->addDependency(tasks->objLocalBounce, {}, {tasks->integration, tasks->objClearLocalForces});
    scheduler->addDependency(tasks->objHaloBounce, {}, {tasks->integration, tasks->objHaloFinalFinalize, tasks->objClearHaloForces});

    scheduler->addDependency(tasks->pluginsAfterIntegration, {tasks->objLocalBounce, tasks->objHaloBounce}, {tasks->integration, tasks->wallBounce});

    scheduler->addDependency(tasks->pluginsBeforeParticlesDistribution, {},
                             {tasks->integration, tasks->wallBounce, tasks->objLocalBounce, tasks->objHaloBounce, tasks->pluginsAfterIntegration});
    scheduler->addDependency(tasks->partRedistributeInit, {}, {tasks->pluginsBeforeParticlesDistribution});
    scheduler->addDependency(tasks->partRedistributeFinalize, {}, {tasks->partRedistributeInit});

    scheduler->addDependency(tasks->objRedistInit, {}, {tasks->integration, tasks->wallBounce, tasks->objReverseFinalFinalize, tasks->pluginsAfterIntegration});
    scheduler->addDependency(tasks->objRedistFinalize, {}, {tasks->objRedistInit});
    scheduler->addDependency(tasks->objClearLocalForces, {tasks->objLocalBounce}, {tasks->integration, tasks->objRedistFinalize});

    scheduler->setHighPriority(tasks->objReverseFinalInit);
    scheduler->setHighPriority(tasks->partHaloIntermediateInit);
    scheduler->setHighPriority(tasks->partHaloIntermediateFinalize);
    scheduler->setHighPriority(tasks->objHaloIntermediateInit);
    scheduler->setHighPriority(tasks->objHaloIntermediateFinalize);
    scheduler->setHighPriority(tasks->objClearHaloIntermediate);
    scheduler->setHighPriority(tasks->objReverseFinalInit);
    scheduler->setHighPriority(tasks->objReverseFinalFinalize);
    scheduler->setHighPriority(tasks->haloIntermediate);
    scheduler->setHighPriority(tasks->partHaloFinalInit);
    scheduler->setHighPriority(tasks->partHaloFinalFinalize);
    scheduler->setHighPriority(tasks->haloForces);
    scheduler->setHighPriority(tasks->pluginsSerializeSend);

    scheduler->setHighPriority(tasks->objClearLocalForces);
    scheduler->setHighPriority(tasks->objLocalBounce);
    
    scheduler->compile();
}

void Simulation::init()
{
    info("Simulation initiated");

    prepareCellLists();

    prepareInteractions();
    prepareBouncers();
    prepareWalls();

    interactionsIntermediate_->checkCompatibleWith(*interactionsFinal_);

    CUDA_Check( cudaDeviceSynchronize() );

    preparePlugins();
    prepareEngines();

    info("Time-step is set to %f", getCurrentDt());
    
    createTasks();
    buildDependencies(scheduler_.get(), tasks_.get());
}

void Simulation::run(int nsteps)
{
    // Initial preparation
    scheduler_->forceExec( tasks_->objHaloFinalInit,     defaultStream );
    scheduler_->forceExec( tasks_->objHaloFinalFinalize, defaultStream );
    scheduler_->forceExec( tasks_->objClearHaloForces,   defaultStream );
    scheduler_->forceExec( tasks_->objClearLocalForces,  defaultStream );
    execSplitters();

    MirState::StepType begin = state_->currentStep, end = state_->currentStep + nsteps;

    info("Will run %d iterations now", nsteps);


    for (state_->currentStep = begin; state_->currentStep < end; state_->currentStep++)
    {
        debug("===============================================================================\n"
                "Timestep: %d, simulation time: %f", state_->currentStep, state_->currentTime);

        scheduler_->run();
        
        state_->currentTime += state_->dt;
    }

    // Finish the redistribution by rebuilding the cell-lists
    scheduler_->forceExec( tasks_->cellLists, defaultStream );

    info("Finished with %d iterations", nsteps);
    MPI_Check( MPI_Barrier(cartComm_) );

    for (auto& pl : plugins)
        pl->finalize();

    notifyPostProcess(stoppingTag, stoppingMsg);
}

void Simulation::notifyPostProcess(int tag, int msg) const
{
    if (interComm_ != MPI_COMM_NULL)
    {
        MPI_Check( MPI_Ssend(&msg, 1, MPI_INT, rank_, tag, interComm_) );
        debug("notify postprocess with tag %d and message %d", tag, msg);
    }
}

void Simulation::restartState(const std::string& folder)
{
    auto filename = createCheckpointName(folder, "state", "txt");
    auto good = TextIO::read(filename, state_->currentTime, state_->currentStep, checkpointId_);
    if (!good) die("failed to read '%s'\n", filename.c_str());    
}

void Simulation::checkpointState()
{
    auto filename = createCheckpointNameWithId(checkpointInfo_.folder, "state", "txt", checkpointId_);

    if (rank_ == 0)
        TextIO::write(filename, state_->currentTime, state_->currentStep, checkpointId_);

    createCheckpointSymlink(cartComm_, checkpointInfo_.folder, "state", "txt", checkpointId_);
}

static void advanceCheckpointId(int& checkpointId, CheckpointIdAdvanceMode mode)
{
    if (mode == CheckpointIdAdvanceMode::PingPong)
        checkpointId = checkpointId xor 1;
    else
        ++checkpointId;
}

void Simulation::restart(const std::string& folder)
{
    this->restartState(folder);

    CUDA_Check( cudaDeviceSynchronize() );

    info("Reading simulation state, from folder %s", folder.c_str());

    for (auto& pv : particleVectors_)
        pv->restart(cartComm_, folder);

    for (auto& handler : bouncerMap_)
        handler.second->restart(cartComm_, folder);

    for (auto& handler : integratorMap_)
        handler.second->restart(cartComm_, folder);

    for (auto& handler : interactionMap_)
        handler.second->restart(cartComm_, folder);

    for (auto& handler : wallMap_)
        handler.second->restart(cartComm_, folder);

    for (auto& handler : belongingCheckerMap_)
        handler.second->restart(cartComm_, folder);

    for (auto& handler : plugins)
        handler->restart(cartComm_, folder);

    CUDA_Check( cudaDeviceSynchronize() );

    // advance checkpoint Id so that next checkpoint does not override this one
    advanceCheckpointId(checkpointId_, checkpointInfo_.mode);
}

void Simulation::checkpoint()
{
    this->checkpointState();
    
    CUDA_Check( cudaDeviceSynchronize() );
    
    info("Writing simulation state, into folder %s", checkpointInfo_.folder.c_str());
    
    for (auto& pv : particleVectors_)
        pv->checkpoint(cartComm_, checkpointInfo_.folder, checkpointId_);
    
    for (auto& handler : bouncerMap_)
        handler.second->checkpoint(cartComm_, checkpointInfo_.folder, checkpointId_);
    
    for (auto& handler : integratorMap_)
        handler.second->checkpoint(cartComm_, checkpointInfo_.folder, checkpointId_);
    
    for (auto& handler : interactionMap_)
        handler.second->checkpoint(cartComm_, checkpointInfo_.folder, checkpointId_);
    
    for (auto& handler : wallMap_)
        handler.second->checkpoint(cartComm_, checkpointInfo_.folder, checkpointId_);
    
    for (auto& handler : belongingCheckerMap_)
        handler.second->checkpoint(cartComm_, checkpointInfo_.folder, checkpointId_);
    
    for (auto& handler : plugins)
        handler->checkpoint(cartComm_, checkpointInfo_.folder, checkpointId_);

    advanceCheckpointId(checkpointId_, checkpointInfo_.mode);

    notifyPostProcess(checkpointTag, checkpointId_);
    
    CUDA_Check( cudaDeviceSynchronize() );
}

MIRHEO_MEMBER_VARS_2(Simulation::IntegratorPrototype, pv, integrator);
MIRHEO_MEMBER_VARS_4(Simulation::InteractionPrototype, rc, pv1, pv2, interaction);
MIRHEO_MEMBER_VARS_3(Simulation::WallPrototype, wall, pv, maximumPartTravel);
MIRHEO_MEMBER_VARS_2(Simulation::CheckWallPrototype, wall, every);
MIRHEO_MEMBER_VARS_2(Simulation::BouncerPrototype, bouncer, pv);
MIRHEO_MEMBER_VARS_4(Simulation::BelongingCorrectionPrototype, checker, pvIn, pvOut, every);
MIRHEO_MEMBER_VARS_4(Simulation::SplitterPrototype, checker, pvSrc, pvIn, pvOut);

void Simulation::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<Simulation>(this, _saveSnapshot(saver, "Simulation"));
}

ConfigObject Simulation::_saveSnapshot(Saver& saver, const std::string &typeName)
{
    ConfigObject config = MirObject::_saveSnapshot(saver, "Simulation", typeName);
    config.emplace("checkpointId",        saver(checkpointId_));
    config.emplace("checkpointInfo",      saver(checkpointInfo_));

    config.emplace("particleVectors",     saver(particleVectors_));

    config.emplace("bouncerMap",          saver(bouncerMap_));
    config.emplace("integratorMap",       saver(integratorMap_));
    config.emplace("interactionMap",      saver(interactionMap_));
    config.emplace("wallMap",             saver(wallMap_));
    config.emplace("belongingCheckerMap", saver(belongingCheckerMap_));

    config.emplace("plugins",             saver(plugins));

    config.emplace("integratorPrototypes",          saver(integratorPrototypes_));
    config.emplace("interactionPrototypes",         saver(interactionPrototypes_));
    config.emplace("wallPrototypes",                saver(wallPrototypes_));
    config.emplace("checkWallPrototypes",           saver(checkWallPrototypes_));
    config.emplace("bouncerPrototypes",             saver(bouncerPrototypes_));
    config.emplace("belongingCorrectionPrototypes", saver(belongingCorrectionPrototypes_));
    config.emplace("splitterPrototypes",            saver(splitterPrototypes_));

    config.emplace("pvsIntegratorMap",    saver(pvsIntegratorMap_));
    return config;
}

void Simulation::saveDependencyGraph_GraphML(const std::string& fname, bool current) const
{
    if (rank_ != 0) return;

    if (current)
    {
        scheduler_->saveDependencyGraph_GraphML(fname);
    }
    else
    {
        TaskScheduler s;
        SimulationTasks t;
        
        createTasksDummy(&s, &t);
        buildDependencies(&s, &t);

        s.saveDependencyGraph_GraphML(fname);
    }
}

} // namespace mirheo
