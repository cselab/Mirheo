#include "simulation.h"

#include <core/bouncers/interface.h>
#include <core/celllist.h>
#include <core/initial_conditions/interface.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/managers/interactions.h>
#include <core/mpi/api.h>
#include <core/object_belonging/interface.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/task_scheduler.h>
#include <core/utils/folders.h>
#include <core/utils/make_unique.h>
#include <core/utils/restart_helpers.h>
#include <core/walls/interface.h>
#include <core/ymero_state.h>
#include <plugins/interface.h>

#include <algorithm>
#include <cuda_profiler_api.h>

Simulation::Simulation(const MPI_Comm &cartComm, const MPI_Comm &interComm, YmrState *state,
                       int globalCheckpointEvery, std::string checkpointFolder,
                       bool gpuAwareMPI)
    : nranks3D(nranks3D),
      interComm(interComm),
      state(state),
      globalCheckpointEvery(globalCheckpointEvery),
      checkpointFolder(checkpointFolder),
      gpuAwareMPI(gpuAwareMPI),
      scheduler(std::make_unique<TaskScheduler>()),
      interactionManager(std::make_unique<InteractionManager>())
{
    int nranks[3], periods[3], coords[3];

    MPI_Check(MPI_Comm_dup(cartComm, &this->cartComm));
    MPI_Check(MPI_Cart_get(cartComm, 3, nranks, periods, coords));
    MPI_Check(MPI_Comm_rank(cartComm, &rank));

    nranks3D = {nranks[0], nranks[1], nranks[2]};
    rank3D   = {coords[0], coords[1], coords[2]};

    createFoldersCollective(cartComm, checkpointFolder);

    state->reinitTime();
    
    info("Simulation initialized, subdomain size is [%f %f %f], subdomain starts "
         "at [%f %f %f]",
         state->domain.localSize.x, state->domain.localSize.y, state->domain.localSize.z,
         state->domain.globalStart.x, state->domain.globalStart.y, state->domain.globalStart.z);    
}

Simulation::~Simulation()
{
    MPI_Check( MPI_Comm_free(&cartComm) );
}


//================================================================================================
// Access for plugins
//================================================================================================


std::vector<ParticleVector*> Simulation::getParticleVectors() const
{
    std::vector<ParticleVector*> res;
    for (auto& pv : particleVectors)
        res.push_back(pv.get());

    return res;
}

ParticleVector* Simulation::getPVbyName(std::string name) const
{
    auto pvIt = pvIdMap.find(name);
    return (pvIt != pvIdMap.end()) ? particleVectors[pvIt->second].get() : nullptr;
}

std::shared_ptr<ParticleVector> Simulation::getSharedPVbyName(std::string name) const
{
    auto pvIt = pvIdMap.find(name);
    return (pvIt != pvIdMap.end()) ? particleVectors[pvIt->second] : std::shared_ptr<ParticleVector>(nullptr);
}

ParticleVector* Simulation::getPVbyNameOrDie(std::string name) const
{
    auto pv = getPVbyName(name);
    if (pv == nullptr)
        die("No such particle vector: %s", name.c_str());
    return pv;
}

ObjectVector* Simulation::getOVbyNameOrDie(std::string name) const
{
    auto pv = getPVbyName(name);
    auto ov = dynamic_cast<ObjectVector*>(pv);
    if (pv == nullptr)
        die("No such particle vector: %s", name.c_str());
    return ov;
}

Wall* Simulation::getWallByNameOrDie(std::string name) const
{
    if (wallMap.find(name) == wallMap.end())
        die("No such wall: %s", name.c_str());

    auto it = wallMap.find(name);
    return it->second.get();
}

CellList* Simulation::gelCellList(ParticleVector* pv) const
{
    auto clvecIt = cellListMap.find(pv);
    if (clvecIt == cellListMap.end())
        die("Particle Vector '%s' is not registered or broken", pv->name.c_str());

    if (clvecIt->second.size() == 0)
        return nullptr;
    else
        return clvecIt->second[0].get();
}

MPI_Comm Simulation::getCartComm() const
{
    return cartComm;
}

float Simulation::getCurrentDt() const
{
    return state->dt;
}

float Simulation::getCurrentTime() const
{
    return state->currentTime;
}

float Simulation::getMaxEffectiveCutoff() const
{
    return interactionManager->getMaxEffectiveCutoff();
}

void Simulation::saveDependencyGraph_GraphML(std::string fname) const
{
    if (rank == 0)
        scheduler->saveDependencyGraph_GraphML(fname);
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

void Simulation::registerParticleVector(std::shared_ptr<ParticleVector> pv, std::shared_ptr<InitialConditions> ic, int checkpointEvery)
{
    std::string name = pv->name;

    if (name == "none" || name == "all" || name == "")
        die("Invalid name for a particle vector (reserved word or empty): '%s'", name.c_str());
    
    if (pv->name.rfind("_", 0) == 0)
        die("Identifier of Particle Vectors cannot start with _");

    if (pvIdMap.find(name) != pvIdMap.end())
        die("More than one particle vector is called %s", name.c_str());

    if (restartStatus != RestartStatus::Anew)
        pv->restart(cartComm, restartFolder);
    else
    {
        if (ic != nullptr)
            ic->exec(cartComm, pv.get(), 0);
    }

    pvsCheckPointPrototype.push_back({pv.get(), checkpointEvery});

    auto ov = dynamic_cast<ObjectVector*>(pv.get());
    if(ov != nullptr)
    {
        info("Registered object vector '%s', %d objects, %d particles", name.c_str(), ov->local()->nObjects, ov->local()->size());
        objectVectors.push_back(ov);
    }
    else
        info("Registered particle vector '%s', %d particles", name.c_str(), pv->local()->size());

    particleVectors.push_back(std::move(pv));
    pvIdMap[name] = particleVectors.size() - 1;
}

void Simulation::registerWall(std::shared_ptr<Wall> wall, int every)
{
    std::string name = wall->name;

    if (wallMap.find(name) != wallMap.end())
        die("More than one wall is called %s", name.c_str());

    checkWallPrototypes.push_back({wall.get(), every});

    // Let the wall know the particle vector associated with it
    wall->setup(cartComm);
    if (restartStatus != RestartStatus::Anew)
        wall->restart(cartComm, restartFolder);

    info("Registered wall '%s'", name.c_str());

    wallMap[name] = std::move(wall);
}

void Simulation::registerInteraction(std::shared_ptr<Interaction> interaction)
{
    std::string name = interaction->name;
    if (interactionMap.find(name) != interactionMap.end())
        die("More than one interaction is called %s", name.c_str());

    if (restartStatus != RestartStatus::Anew)
        interaction->restart(cartComm, restartFolder);

    interactionMap[name] = std::move(interaction);
}

void Simulation::registerIntegrator(std::shared_ptr<Integrator> integrator)
{
    std::string name = integrator->name;
    if (integratorMap.find(name) != integratorMap.end())
        die("More than one integrator is called %s", name.c_str());

    if (restartStatus != RestartStatus::Anew)
        integrator->restart(cartComm, restartFolder);
    
    integratorMap[name] = std::move(integrator);
}

void Simulation::registerBouncer(std::shared_ptr<Bouncer> bouncer)
{
    std::string name = bouncer->name;
    if (bouncerMap.find(name) != bouncerMap.end())
        die("More than one bouncer is called %s", name.c_str());

    if (restartStatus != RestartStatus::Anew)
        bouncer->restart(cartComm, restartFolder);
    
    bouncerMap[name] = std::move(bouncer);
}

void Simulation::registerObjectBelongingChecker(std::shared_ptr<ObjectBelongingChecker> checker)
{
    std::string name = checker->name;
    if (belongingCheckerMap.find(name) != belongingCheckerMap.end())
        die("More than one splitter is called %s", name.c_str());

    if (restartStatus != RestartStatus::Anew)
        checker->restart(cartComm, restartFolder);
    
    belongingCheckerMap[name] = std::move(checker);
}

void Simulation::registerPlugin(std::shared_ptr<SimulationPlugin> plugin)
{
    std::string name = plugin->name;

    bool found = false;
    for (auto& pl : plugins)
        if (pl->name == name) found = true;

    if (found)
        die("More than one plugin is called %s", name.c_str());

    if (restartStatus != RestartStatus::Anew)
        plugin->restart(cartComm, restartFolder);
    
    plugins.push_back(std::move(plugin));
}

//================================================================================================
// Applying something to something else
//================================================================================================

void Simulation::setIntegrator(std::string integratorName, std::string pvName)
{
    if (integratorMap.find(integratorName) == integratorMap.end())
        die("No such integrator: %s", integratorName.c_str());
    auto integrator = integratorMap[integratorName].get();

    auto pv = getPVbyNameOrDie(pvName);

    if (pvsIntegratorMap.find(pvName) != pvsIntegratorMap.end())
        die("particle vector '%s' already set to integrator '%s'",
            pvName.c_str(), pvsIntegratorMap[pvName].c_str());

    pvsIntegratorMap[pvName] = integratorName;
    
    integrator->setPrerequisites(pv);

    integratorsStage1.push_back([integrator, pv] (cudaStream_t stream) {
        integrator->stage1(pv, stream);
    });

    integratorsStage2.push_back([integrator, pv] (cudaStream_t stream) {
        integrator->stage2(pv, stream);
    });
}

void Simulation::setInteraction(std::string interactionName, std::string pv1Name, std::string pv2Name)
{
    auto pv1 = getPVbyNameOrDie(pv1Name);
    auto pv2 = getPVbyNameOrDie(pv2Name);

    if (interactionMap.find(interactionName) == interactionMap.end())
        die("No such interaction: %s", interactionName.c_str());
    auto interaction = interactionMap[interactionName].get();    

    float rc = interaction->rc;
    interactionPrototypes.push_back({rc, pv1, pv2, interaction});
}

void Simulation::setBouncer(std::string bouncerName, std::string objName, std::string pvName)
{
    auto pv = getPVbyNameOrDie(pvName);

    auto ov = dynamic_cast<ObjectVector*> (getPVbyName(objName));
    if (ov == nullptr)
        die("No such object vector: %s", objName.c_str());

    if (bouncerMap.find(bouncerName) == bouncerMap.end())
        die("No such bouncer: %s", bouncerName.c_str());
    auto bouncer = bouncerMap[bouncerName].get();

    bouncer->setup(ov);
    bouncer->setPrerequisites(pv);
    bouncerPrototypes.push_back({bouncer, pv});
}

void Simulation::setWallBounce(std::string wallName, std::string pvName)
{
    auto pv = getPVbyNameOrDie(pvName);

    if (wallMap.find(wallName) == wallMap.end())
        die("No such wall: %s", wallName.c_str());
    auto wall = wallMap[wallName].get();

    wall->setPrerequisites(pv);
    wallPrototypes.push_back( {wall, pv} );
}

void Simulation::setObjectBelongingChecker(std::string checkerName, std::string objName)
{
    auto ov = dynamic_cast<ObjectVector*>(getPVbyNameOrDie(objName));
    if (ov == nullptr)
        die("No such object vector %s", objName.c_str());

    if (belongingCheckerMap.find(checkerName) == belongingCheckerMap.end())
        die("No such belonging checker: %s", checkerName.c_str());
    auto checker = belongingCheckerMap[checkerName].get();

    // TODO: do this normal'no blyat!
    checker->setup(ov);
}

//
//
//

void Simulation::applyObjectBelongingChecker(std::string checkerName,
            std::string source, std::string inside, std::string outside,
            int checkEvery, int checkpointEvery)
{
    auto pvSource = getPVbyNameOrDie(source);

    if (inside == outside)
        die("Splitting into same pvs: %s into %s %s",
                source.c_str(), inside.c_str(), outside.c_str());

    if (source != inside && source != outside)
        die("At least one of the split destinations should be the same as source: %s into %s %s",
                source.c_str(), inside.c_str(), outside.c_str());

    if (belongingCheckerMap.find(checkerName) == belongingCheckerMap.end())
        die("No such belonging checker: %s", checkerName.c_str());

    if (getPVbyName(inside) != nullptr && inside != source)
        die("Cannot split into existing particle vector: %s into %s %s",
                source.c_str(), inside.c_str(), outside.c_str());

    if (getPVbyName(outside) != nullptr && outside != source)
        die("Cannot split into existing particle vector: %s into %s %s",
                source.c_str(), inside.c_str(), outside.c_str());


    auto checker = belongingCheckerMap[checkerName].get();

    std::shared_ptr<ParticleVector> pvInside, pvOutside;

    if (inside != "none" && getPVbyName(inside) == nullptr)
    {
        pvInside = std::make_shared<ParticleVector> (state, inside, pvSource->mass);
        registerParticleVector(pvInside, nullptr, checkpointEvery);
    }

    if (outside != "none" && getPVbyName(outside) == nullptr)
    {
        pvOutside = std::make_shared<ParticleVector> (state, outside, pvSource->mass);
        registerParticleVector(pvOutside, nullptr, checkpointEvery);
    }

    splitterPrototypes.push_back({checker, pvSource, getPVbyName(inside), getPVbyName(outside)});

    belongingCorrectionPrototypes.push_back({checker, getPVbyName(inside), getPVbyName(outside), checkEvery});
}

static void sortDescendingOrder(std::vector<float>& v)
{
    std::sort(v.begin(), v.end(), [] (float a, float b) { return a > b; });
}

// assume sorted array (ascending or descending)
static void removeDuplicatedElements(std::vector<float>& v, float tolerance)
{
    auto it = std::unique(v.begin(), v.end(), [=] (float a, float b) { return fabs(a - b) < tolerance; });
    v.resize( std::distance(v.begin(), it) );    
}

void Simulation::prepareCellLists()
{
    info("Preparing cell-lists");

    std::map<ParticleVector*, std::vector<float>> cutOffMap;

    // Deal with the cell-lists and interactions
    for (auto prototype : interactionPrototypes)
    {
        float rc = prototype.rc;
        cutOffMap[prototype.pv1].push_back(rc);
        cutOffMap[prototype.pv2].push_back(rc);
    }

    for (auto& cutoffPair : cutOffMap)
    {
        auto& pv      = cutoffPair.first;
        auto& cutoffs = cutoffPair.second;

        sortDescendingOrder(cutoffs);
        removeDuplicatedElements(cutoffs, rcTolerance);

        bool primary = true;

        // Don't use primary cell-lists with ObjectVectors
        if (dynamic_cast<ObjectVector*>(pv) != nullptr)
            primary = false;

        for (auto rc : cutoffs)
        {
            cellListMap[pv].push_back(primary ?
                    std::make_unique<PrimaryCellList>(pv, rc, state->domain.localSize) :
                    std::make_unique<CellList>       (pv, rc, state->domain.localSize));
            primary = false;
        }
    }

    for (auto& pv : particleVectors)
    {
        auto pvptr = pv.get();
        if (cellListMap[pvptr].empty())
        {
            const float defaultRc = 1.f;
            bool primary = true;

            // Don't use primary cell-lists with ObjectVectors
            if (dynamic_cast<ObjectVector*>(pvptr) != nullptr)
                primary = false;

            cellListMap[pvptr].push_back
                (primary ?
                 std::make_unique<PrimaryCellList>(pvptr, defaultRc, state->domain.localSize) :
                 std::make_unique<CellList>       (pvptr, defaultRc, state->domain.localSize));
        }
    }
}

// Choose a CL with smallest but bigger than rc cell
static CellList* selectBestClist(std::vector<std::unique_ptr<CellList>>& cellLists, float rc, float tolerance)
{
    float minDiff = 1e6;
    CellList* best = nullptr;
    
    for (auto& cl : cellLists) {
        float diff = cl->rc - rc;
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

    for (auto& prototype : interactionPrototypes)
    {
        auto  rc = prototype.rc;
        auto pv1 = prototype.pv1;
        auto pv2 = prototype.pv2;

        auto& clVec1 = cellListMap[pv1];
        auto& clVec2 = cellListMap[pv2];

        CellList *cl1, *cl2;

        cl1 = selectBestClist(clVec1, rc, rcTolerance);
        cl2 = selectBestClist(clVec2, rc, rcTolerance);
        
        auto inter = prototype.interaction;

        inter->setPrerequisites(pv1, pv2, cl1, cl2);

        interactionManager->add(inter, pv1, pv2, cl1, cl2);
    }
}

void Simulation::prepareBouncers()
{
    info("Preparing object bouncers");

    for (auto& prototype : bouncerPrototypes)
    {
        auto bouncer = prototype.bouncer;
        auto pv      = prototype.pv;

        if (pvsIntegratorMap.find(pv->name) == pvsIntegratorMap.end())
            die("Setting bouncer '%s': particle vector '%s' has no integrator, required for bounce back",
                bouncer->name.c_str(), pv->name.c_str());
        
        auto& clVec = cellListMap[pv];

        if (clVec.empty()) continue;

        CellList *cl = clVec[0].get();

        regularBouncers.push_back([bouncer, pv, cl] (cudaStream_t stream) {
            bouncer->bounceLocal(pv, cl, stream);
        });

        haloBouncers.   push_back([bouncer, pv, cl] (cudaStream_t stream) {
            bouncer->bounceHalo (pv, cl, stream);
        });
    }
}

void Simulation::prepareWalls()
{
    info("Preparing walls");

    for (auto& prototype : wallPrototypes)
    {
        auto wall = prototype.wall;
        auto pv   = prototype.pv;

        auto& clVec = cellListMap[pv];

        if (clVec.empty()) continue;

        CellList *cl = clVec[0].get();

        wall->attach(pv, cl);
    }

    for (auto& wall : wallMap)
    {
        auto wallPtr = wall.second.get();

        // All the particles should be removed from within the wall,
        // even those that do not interact with it
        // Only frozen wall particles will remain
        for (auto& anypv : particleVectors)
            wallPtr->removeInner(anypv.get());
    }
}

void Simulation::preparePlugins()
{
    info("Preparing plugins");
    for (auto& pl : plugins) {
        debug("Setup and handshake of plugin %s", pl->name.c_str());
        pl->setup(this, cartComm, interComm);
        pl->handshake();
    }
    info("done Preparing plugins");
}


void Simulation::prepareEngines()
{
    auto redistImp              = std::make_unique<ParticleRedistributor>();
    auto haloImp                = std::make_unique<ParticleHaloExchanger>();
    auto haloIntermediateImp    = std::make_unique<ParticleHaloExchanger>();
    auto objRedistImp           = std::make_unique<ObjectRedistributor>();
    auto objHaloImp             = std::make_unique<ObjectHaloExchanger>();
    auto objHaloIntermediateImp = std::make_unique<ObjectHaloExchanger>();
    auto objForcesImp           = std::make_unique<ObjectForcesReverseExchanger>(objHaloImp.get());

    debug("Attaching particle vectors to halo exchanger and redistributor");
    for (auto& pv : particleVectors)
    {
        auto  pvPtr       = pv.get();
        auto& cellListVec = cellListMap[pvPtr];        

        if (cellListVec.size() == 0) continue;

        CellList *clInt = interactionManager->getLargestCellListNeededForIntermediate(pvPtr);
        CellList *clOut = interactionManager->getLargestCellListNeededForFinal(pvPtr);

        auto extraInt = interactionManager->getExtraIntermediateChannels(pvPtr);
        // auto extraOut = interactionManager->getExtraFinalChannels(pvPtr); // TODO: for reverse exchanger

        auto cl = cellListVec[0].get();
        auto ov = dynamic_cast<ObjectVector*>(pvPtr);
        
        if (ov == nullptr) {
            redistImp->attach(pvPtr, cl);
            
            if (clInt != nullptr)
                haloIntermediateImp->attach(pvPtr, clInt, {});

            if (clOut != nullptr)
                haloImp->attach(pvPtr, clOut, extraInt);            
        }
        else {
            objRedistImp->attach(ov);

            if (clInt != nullptr)
                objHaloIntermediateImp->attach(ov, clInt->rc, {});

            auto extraToExchange = extraInt;
            
            for (auto& entry : bouncerMap)
            {
                auto& bouncer = entry.second;
                if (bouncer->getObjectVector() == ov)
                {
                    auto extraChannels = bouncer->getChannelsToBeExchanged();
                    std::copy(extraChannels.begin(), extraChannels.end(),
                              std::back_inserter(extraToExchange));
                }
            }

            objHaloImp  ->attach(ov, cl->rc, extraToExchange); // always active because of bounce back; TODO: check if bounce back is active
            objForcesImp->attach(ov);
        }
    }
    
    std::function< std::unique_ptr<ExchangeEngine>(std::unique_ptr<ParticleExchanger>) > makeEngine;
    
    // If we're on one node, use a singleNode engine
    // otherwise use MPI
    if (nranks3D.x * nranks3D.y * nranks3D.z == 1)
        makeEngine = [this] (std::unique_ptr<ParticleExchanger> exch) {
            return std::make_unique<SingleNodeEngine> (std::move(exch));
        };
    else
        makeEngine = [this] (std::unique_ptr<ParticleExchanger> exch) {
            return std::make_unique<MPIExchangeEngine> (std::move(exch), cartComm, gpuAwareMPI);
        };
    
    redistributor       = makeEngine(std::move(redistImp));
    halo                = makeEngine(std::move(haloImp));
    haloIntermediate    = makeEngine(std::move(haloIntermediateImp));
    objRedistibutor     = makeEngine(std::move(objRedistImp));
    objHalo             = makeEngine(std::move(objHaloImp));
    objHaloIntermediate = makeEngine(std::move(objHaloIntermediateImp));
    objHaloForces       = makeEngine(std::move(objForcesImp));
}

void Simulation::execSplitters()
{
    info("Splitting particle vectors with respect to object belonging");

    for (auto& prototype : splitterPrototypes)
    {
        auto checker = prototype.checker;
        auto src     = prototype.pvSrc;
        auto inside  = prototype.pvIn;
        auto outside = prototype.pvOut;

        checker->splitByBelonging(src, inside, outside, 0);
    }
}

void Simulation::init()
{
    info("Simulation initiated");

    prepareCellLists();

    prepareInteractions();
    prepareBouncers();
    prepareWalls();

    interactionManager->check();

    CUDA_Check( cudaDeviceSynchronize() );

    preparePlugins();
    prepareEngines();

    assemble();
    
    // Initial preparation
    scheduler->forceExec( scheduler->getTaskId("Object halo init"), 0 );
    scheduler->forceExec( scheduler->getTaskId("Object halo finalize"), 0 );
    scheduler->forceExec( scheduler->getTaskId("Clear object halo forces"), 0 );
    scheduler->forceExec( scheduler->getTaskId("Clear object local forces"), 0 );

    execSplitters();
}

void Simulation::assemble()
{    
    info("Time-step is set to %f", getCurrentDt());

    auto task_checkpoint                          = scheduler->createTask("Checkpoint");
    auto task_cellLists                           = scheduler->createTask("Build cell-lists");
    auto task_integration                         = scheduler->createTask("Integration");

    auto task_clearIntermediate                   = scheduler->createTask("Clear intermediate");
    auto task_haloIntermediateInit                = scheduler->createTask("Halo intermediate init");
    auto task_haloIntermediateFinalize            = scheduler->createTask("Halo intermediate finalize");
    auto task_localIntermediate                   = scheduler->createTask("Local intermediate");
    auto task_haloIntermediate                    = scheduler->createTask("Halo intermediate");
    auto task_accumulateInteractionIntermediate   = scheduler->createTask("Accumulate intermediate");
    auto task_gatherInteractionIntermediate       = scheduler->createTask("Gather intermediate");

    auto task_clearFinalOutput                    = scheduler->createTask("Clear forces");
    auto task_haloInit                            = scheduler->createTask("Halo init");
    auto task_haloFinalize                        = scheduler->createTask("Halo finalize");
    auto task_localForces                         = scheduler->createTask("Local forces");
    auto task_haloForces                          = scheduler->createTask("Halo forces");
    auto task_accumulateInteractionFinal          = scheduler->createTask("Accumulate forces");

    auto task_objHaloInit                         = scheduler->createTask("Object halo init");
    auto task_objHaloFinalize                     = scheduler->createTask("Object halo finalize");
    auto task_objForcesInit                       = scheduler->createTask("Object forces exchange: init");
    auto task_objForcesFinalize                   = scheduler->createTask("Object forces exchange: finalize");
    auto task_clearObjHaloForces                  = scheduler->createTask("Clear object halo forces");
    auto task_clearObjLocalForces                 = scheduler->createTask("Clear object local forces");

    auto task_objLocalBounce                      = scheduler->createTask("Local object bounce");
    auto task_objHaloBounce                       = scheduler->createTask("Halo object bounce");
    auto task_correctObjBelonging                 = scheduler->createTask("Correct object belonging");

    auto task_wallBounce                          = scheduler->createTask("Wall bounce");
    auto task_wallCheck                           = scheduler->createTask("Wall check");    

    auto task_redistributeInit                    = scheduler->createTask("Redistribute init");
    auto task_redistributeFinalize                = scheduler->createTask("Redistribute finalize");
    auto task_objRedistInit                       = scheduler->createTask("Object redistribute init");
    auto task_objRedistFinalize                   = scheduler->createTask("Object redistribute finalize");

    auto task_pluginsBeforeCellLists              = scheduler->createTask("Plugins: before cell lists");
    auto task_pluginsBeforeForces                 = scheduler->createTask("Plugins: before forces");
    auto task_pluginsSerializeSend                = scheduler->createTask("Plugins: serialize and send");
    auto task_pluginsBeforeIntegration            = scheduler->createTask("Plugins: before integration");
    auto task_pluginsAfterIntegration             = scheduler->createTask("Plugins: after integration");
    auto task_pluginsBeforeParticlesDistribution  = scheduler->createTask("Plugins: before particles distribution");


    if (globalCheckpointEvery > 0)
        scheduler->addTask(task_checkpoint,
                           [this](cudaStream_t stream) { this->checkpoint(); },
                           globalCheckpointEvery);

    for (auto prototype : pvsCheckPointPrototype)
        if (prototype.checkpointEvery > 0 && globalCheckpointEvery == 0) {
            info("Will save checkpoint of particle vector '%s' every %d timesteps",
                 prototype.pv->name.c_str(), prototype.checkpointEvery);

            scheduler->addTask( task_checkpoint, [prototype, this] (cudaStream_t stream) {
                prototype.pv->checkpoint(cartComm, checkpointFolder);
            }, prototype.checkpointEvery );
        }


    for (auto& clVec : cellListMap)
        for (auto& cl : clVec.second)
        {
            auto clPtr = cl.get();
            scheduler->addTask(task_cellLists, [clPtr] (cudaStream_t stream) { clPtr->build(stream); } );
        }

    // Only particle forces, not object ones here
    for (auto& pv : particleVectors)
    {
        auto pvPtr = pv.get();
        scheduler->addTask(task_clearFinalOutput,
                           [this, pvPtr] (cudaStream_t stream) { interactionManager->clearIntermediates(pvPtr, stream); } );
        scheduler->addTask(task_clearIntermediate,
                           [this, pvPtr] (cudaStream_t stream) { interactionManager->clearFinal(pvPtr, stream); } );
    }

    for (auto& pl : plugins)
    {
        auto plPtr = pl.get();

        scheduler->addTask(task_pluginsBeforeCellLists, [plPtr, this] (cudaStream_t stream) {
            plPtr->beforeCellLists(stream);
        });

        scheduler->addTask(task_pluginsBeforeForces, [plPtr, this] (cudaStream_t stream) {
            plPtr->beforeForces(stream);
        });

        scheduler->addTask(task_pluginsSerializeSend, [plPtr] (cudaStream_t stream) {
            plPtr->serializeAndSend(stream);
        });

        scheduler->addTask(task_pluginsBeforeIntegration, [plPtr] (cudaStream_t stream) {
            plPtr->beforeIntegration(stream);
        });

        scheduler->addTask(task_pluginsAfterIntegration, [plPtr] (cudaStream_t stream) {
            plPtr->afterIntegration(stream);
        });

        scheduler->addTask(task_pluginsBeforeParticlesDistribution, [plPtr] (cudaStream_t stream) {
            plPtr->beforeParticleDistribution(stream);
        });
    }


    // If we have any non-object vectors
    if (particleVectors.size() != objectVectors.size())
    {
        scheduler->addTask(task_haloIntermediateInit, [this] (cudaStream_t stream) {
            haloIntermediate->init(stream);
        });

        scheduler->addTask(task_haloIntermediateFinalize, [this] (cudaStream_t stream) {
            haloIntermediate->finalize(stream);
        });

        scheduler->addTask(task_haloInit, [this] (cudaStream_t stream) {
            halo->init(stream);
        });

        scheduler->addTask(task_haloFinalize, [this] (cudaStream_t stream) {
            halo->finalize(stream);
        });

        scheduler->addTask(task_redistributeInit, [this] (cudaStream_t stream) {
            redistributor->init(stream);
        });

        scheduler->addTask(task_redistributeFinalize, [this] (cudaStream_t stream) {
            redistributor->finalize(stream);
        });
    }


    scheduler->addTask(task_localIntermediate,
                       [this] (cudaStream_t stream) {
                           interactionManager->executeLocalIntermediate(stream);
                       });

    scheduler->addTask(task_haloIntermediate,
                       [this] (cudaStream_t stream) {
                           interactionManager->executeHaloIntermediate(stream);
                       });

    scheduler->addTask(task_localForces,
                       [this] (cudaStream_t stream) {
                           interactionManager->executeLocalFinal(stream);
                       });

    scheduler->addTask(task_haloForces,
                       [this] (cudaStream_t stream) {
                           interactionManager->executeHaloFinal(stream);
                       });
    

    scheduler->addTask(task_gatherInteractionIntermediate,
                       [this] (cudaStream_t stream) {
                           interactionManager->gatherIntermediate(stream);
                       });

    scheduler->addTask(task_accumulateInteractionIntermediate,
                       [this] (cudaStream_t stream) {
                           interactionManager->accumulateIntermediates(stream);
                       });
            
    scheduler->addTask(task_accumulateInteractionFinal,
                       [this] (cudaStream_t stream) {
                           interactionManager->accumulateFinal(stream);
                       });


    for (auto& integrator : integratorsStage2)
        scheduler->addTask(task_integration, [integrator, this] (cudaStream_t stream) {
            integrator(stream);
        });


    for (auto ov : objectVectors)
        scheduler->addTask(task_clearObjHaloForces, [ov] (cudaStream_t stream) {
            ov->halo()->forces.clear(stream);
        });

    // As there are no primary cell-lists for objects
    // we need to separately clear real obj forces and forces in the cell-lists
    for (auto ovPtr : objectVectors)
    {
        scheduler->addTask(task_clearObjLocalForces, [ovPtr] (cudaStream_t stream) {
            ovPtr->local()->forces.clear(stream);
        });

        scheduler->addTask(task_clearObjLocalForces,
                           [this, ovPtr] (cudaStream_t stream) {
                               interactionManager->clearFinal(ovPtr, stream);
                           });
    }

    for (auto& bouncer : regularBouncers)
        scheduler->addTask(task_objLocalBounce, [bouncer, this] (cudaStream_t stream) {
            bouncer(stream);
    });

    for (auto& bouncer : haloBouncers)
        scheduler->addTask(task_objHaloBounce, [bouncer, this] (cudaStream_t stream) {
            bouncer(stream);
    });

    for (auto& prototype : belongingCorrectionPrototypes)
    {
        auto checker = prototype.checker;
        auto pvIn    = prototype.pvIn;
        auto pvOut   = prototype.pvOut;
        auto every   = prototype.every;

        if (every > 0)
        {
            scheduler->addTask(task_correctObjBelonging, [checker, pvIn, pvOut] (cudaStream_t stream) {
                if (pvIn  != nullptr) checker->splitByBelonging(pvIn,  pvIn, pvOut, stream);
                if (pvOut != nullptr) checker->splitByBelonging(pvOut, pvIn, pvOut, stream);
            }, every);
        }
    }

    if (objectVectors.size() > 0)
    {
        scheduler->addTask(task_objHaloInit, [this] (cudaStream_t stream) {
            objHalo->init(stream);
        });

        scheduler->addTask(task_objHaloFinalize, [this] (cudaStream_t stream) {
            objHalo->finalize(stream);
        });

        scheduler->addTask(task_objForcesInit, [this] (cudaStream_t stream) {
            objHaloForces->init(stream);
        });

        scheduler->addTask(task_objForcesFinalize, [this] (cudaStream_t stream) {
            objHaloForces->finalize(stream);
        });

        scheduler->addTask(task_objRedistInit, [this] (cudaStream_t stream) {
            objRedistibutor->init(stream);
        });

        scheduler->addTask(task_objRedistFinalize, [this] (cudaStream_t stream) {
            objRedistibutor->finalize(stream);
        });
    }

    for (auto& wall : wallMap)
    {
        auto wallPtr = wall.second.get();
        scheduler->addTask(task_wallBounce, [wallPtr, this] (cudaStream_t stream) {    
            wallPtr->bounce(stream);
        });
    }

    for (auto& prototype : checkWallPrototypes)
    {
        auto wall  = prototype.wall;
        auto every = prototype.every;

        if (every > 0)
            scheduler->addTask(task_wallCheck, [this, wall] (cudaStream_t stream) { wall->check(stream); }, every);
    }
    
    scheduler->addDependency(task_pluginsBeforeCellLists, { task_cellLists }, {});
    
    scheduler->addDependency(task_checkpoint, { task_clearFinalOutput }, { task_cellLists });

    scheduler->addDependency(task_correctObjBelonging, { task_cellLists }, {});

    scheduler->addDependency(task_cellLists, {task_clearFinalOutput, task_clearIntermediate}, {});

    
    scheduler->addDependency(task_pluginsBeforeForces, {task_localForces, task_haloForces}, {task_clearFinalOutput});
    scheduler->addDependency(task_pluginsSerializeSend, {task_pluginsBeforeIntegration, task_pluginsAfterIntegration}, {task_pluginsBeforeForces});

    scheduler->addDependency(task_clearObjHaloForces, {task_objHaloBounce}, {task_objHaloFinalize});

    scheduler->addDependency(task_objForcesInit, {}, {task_haloForces});
    scheduler->addDependency(task_objForcesFinalize, {task_accumulateInteractionFinal}, {task_objForcesInit});

    scheduler->addDependency(task_localIntermediate, {}, {task_clearIntermediate});
    scheduler->addDependency(task_haloIntermediateInit, {}, {task_clearIntermediate, task_cellLists});
    scheduler->addDependency(task_haloIntermediateFinalize, {}, {task_haloIntermediateInit});
    scheduler->addDependency(task_haloIntermediate, {}, {task_haloIntermediateFinalize});
    scheduler->addDependency(task_accumulateInteractionIntermediate, {}, {task_localIntermediate, task_haloIntermediate});
    scheduler->addDependency(task_gatherInteractionIntermediate, {}, {task_accumulateInteractionIntermediate});

    scheduler->addDependency(task_localForces, {}, {task_gatherInteractionIntermediate});
    scheduler->addDependency(task_haloInit, {}, {task_pluginsBeforeForces, task_gatherInteractionIntermediate, task_cellLists});
    scheduler->addDependency(task_haloFinalize, {}, {task_haloInit});
    scheduler->addDependency(task_haloForces, {}, {task_haloFinalize});
    scheduler->addDependency(task_accumulateInteractionFinal, {task_integration}, {task_haloForces, task_localForces});

    scheduler->addDependency(task_pluginsBeforeIntegration, {task_integration}, {task_accumulateInteractionFinal});
    scheduler->addDependency(task_wallBounce, {}, {task_integration});
    scheduler->addDependency(task_wallCheck, {task_redistributeInit}, {task_wallBounce});

    scheduler->addDependency(task_objHaloInit, {}, {task_integration, task_objRedistFinalize});
    scheduler->addDependency(task_objHaloFinalize, {}, {task_objHaloInit});

    scheduler->addDependency(task_objLocalBounce, {task_objHaloFinalize}, {task_integration, task_clearObjLocalForces});
    scheduler->addDependency(task_objHaloBounce, {}, {task_integration, task_objHaloFinalize, task_clearObjHaloForces});

    scheduler->addDependency(task_pluginsAfterIntegration, {task_objLocalBounce, task_objHaloBounce}, {task_integration, task_wallBounce});

    scheduler->addDependency(task_pluginsBeforeParticlesDistribution, {},
                             {task_integration, task_wallBounce, task_objLocalBounce, task_objHaloBounce, task_pluginsAfterIntegration});
    scheduler->addDependency(task_redistributeInit, {}, {task_pluginsBeforeParticlesDistribution});
    scheduler->addDependency(task_redistributeFinalize, {}, {task_redistributeInit});

    scheduler->addDependency(task_objRedistInit, {}, {task_integration, task_wallBounce, task_objForcesFinalize, task_pluginsAfterIntegration});
    scheduler->addDependency(task_objRedistFinalize, {}, {task_objRedistInit});
    scheduler->addDependency(task_clearObjLocalForces, {task_objLocalBounce}, {task_integration, task_objRedistFinalize});

    scheduler->setHighPriority(task_objForcesInit);
    scheduler->setHighPriority(task_haloIntermediateInit);
    scheduler->setHighPriority(task_haloIntermediateFinalize);
    scheduler->setHighPriority(task_haloIntermediate);
    scheduler->setHighPriority(task_haloInit);
    scheduler->setHighPriority(task_haloFinalize);
    scheduler->setHighPriority(task_haloForces);
    scheduler->setHighPriority(task_pluginsSerializeSend);

    scheduler->setHighPriority(task_clearObjLocalForces);
    scheduler->setHighPriority(task_objLocalBounce);
    
    scheduler->compile();
}

void Simulation::run(int nsteps)
{
    int begin = state->currentStep, end = state->currentStep + nsteps;

    info("Will run %d iterations now", nsteps);


    for (state->currentStep = begin; state->currentStep < end; state->currentStep++)
    {
        debug("===============================================================================\n"
                "Timestep: %d, simulation time: %f", state->currentStep, state->currentTime);

        scheduler->run();
        
        state->currentTime += state->dt;
    }

    // Finish the redistribution by rebuilding the cell-lists
    scheduler->forceExec( scheduler->getTaskId("Build cell-lists"), 0 );

    info("Finished with %d iterations", nsteps);
    MPI_Check( MPI_Barrier(cartComm) );

    for (auto& pl : plugins)
        pl->finalize();

    if (interComm != MPI_COMM_NULL)
    {
        int dummy = -1;
        int tag = 424242;

        MPI_Check( MPI_Send(&dummy, 1, MPI_INT, rank, tag, interComm) );
        debug("Sending stopping message to the postprocess");
    }
}


void Simulation::restart(std::string folder)
{
//    bool beginning =  particleVectors    .empty() &&
//                      wallMap            .empty() &&
//                      interactionMap     .empty() &&
//                      integratorMap      .empty() &&
//                      bouncerMap         .empty() &&
//                      belongingCheckerMap.empty() &&
//                      plugins            .empty();
//
//    if (!beginning)
//        die("Tried to restart partially initialized simulation! Please only call restart() before registering anything");
//
//    restartStatus = RestartStatus::RestartStrict;
//    restartFolder = folder;
//
//    TextIO::read(folder + "_simulation.state", state->currentTime, state->currentStep);

	TextIO::read(folder + "_simulation.state", state->currentTime, state->currentStep);
	restartFolder = folder;

	CUDA_Check( cudaDeviceSynchronize() );

	info("Reading simulation state, from folder %s", restartFolder.c_str());

	for (auto& pv : particleVectors)
		pv->restart(cartComm, restartFolder);

	for (auto& handler : bouncerMap)
		handler.second->restart(cartComm, restartFolder);

	for (auto& handler : integratorMap)
		handler.second->restart(cartComm, restartFolder);

	for (auto& handler : interactionMap)
		handler.second->restart(cartComm, restartFolder);

	for (auto& handler : wallMap)
		handler.second->restart(cartComm, restartFolder);

	for (auto& handler : belongingCheckerMap)
		handler.second->restart(cartComm, restartFolder);

	for (auto& handler : plugins)
		handler->restart(cartComm, restartFolder);

	CUDA_Check( cudaDeviceSynchronize() );
}

void Simulation::checkpoint()
{
    if (rank == 0)
        TextIO::write(checkpointFolder + "_simulation.state", state->currentTime, state->currentStep);

    CUDA_Check( cudaDeviceSynchronize() );
    
    info("Writing simulation state, into folder %s", checkpointFolder.c_str());
        
    for (auto& pv : particleVectors)
        pv->checkpoint(cartComm, checkpointFolder);
    
    for (auto& handler : bouncerMap)
        handler.second->checkpoint(cartComm, checkpointFolder);
    
    for (auto& handler : integratorMap)
        handler.second->checkpoint(cartComm, checkpointFolder);
    
    for (auto& handler : interactionMap)
        handler.second->checkpoint(cartComm, checkpointFolder);
    
    for (auto& handler : wallMap)
        handler.second->checkpoint(cartComm, checkpointFolder);
    
    for (auto& handler : belongingCheckerMap)
        handler.second->checkpoint(cartComm, checkpointFolder);
    
    for (auto& handler : plugins)
        handler->checkpoint(cartComm, checkpointFolder);
    
    CUDA_Check( cudaDeviceSynchronize() );
}





