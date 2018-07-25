#include "simulation.h"

#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

#include <core/bouncers/interface.h>
#include <core/initial_conditions/interface.h>
#include <core/integrators/interface.h>
#include <core/interactions/interface.h>
#include <core/walls/interface.h>
#include <core/object_belonging/interface.h>
#include <plugins/interface.h>

#include <core/task_scheduler.h>
#include <core/mpi/api.h>

#include <core/utils/folders.h>
#include <core/utils/make_unique.h>

#include <algorithm>

Simulation::Simulation(int3 nranks3D, float3 globalDomainSize, const MPI_Comm& comm, const MPI_Comm& interComm,
                       bool gpuAwareMPI, bool performCleanup) :
nranks3D(nranks3D), interComm(interComm), currentTime(0), currentStep(0), gpuAwareMPI(gpuAwareMPI), cleanup(performCleanup)
{
    int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
    int periods[] = {1, 1, 1};
    int coords[3];

    MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 1, &cartComm) );
    MPI_Check( MPI_Cart_get(cartComm, 3, ranksArr, periods, coords) );
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    rank3D = {coords[0], coords[1], coords[2]};

    domain.globalSize = globalDomainSize;
    domain.localSize = domain.globalSize / make_float3(nranks3D);
    domain.globalStart = {domain.localSize.x * coords[0], domain.localSize.y * coords[1], domain.localSize.z * coords[2]};

    restartFolder  = "./restart/";
    createFoldersCollective(cartComm, restartFolder);

    info("Simulation initialized, subdomain size is [%f %f %f], subdomain starts at [%f %f %f]",
            domain.localSize.x,  domain.localSize.y,  domain.localSize.z,
            domain.globalStart.x, domain.globalStart.y, domain.globalStart.z);

    scheduler = std::make_unique<TaskScheduler>();
    scheduler->createTask("Checkpoint");
}

Simulation::~Simulation() = default;


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


//================================================================================================
// Registration
//================================================================================================

void Simulation::registerParticleVector(std::shared_ptr<ParticleVector> pv, std::shared_ptr<InitialConditions> ic, int checkpointEvery)
{
    std::string name = pv->name;

    if (name == "none" || name == "all" || name == "")
        die("Invalid name for a particle vector (reserved word or empty): '%s'", name.c_str());

    if (pvIdMap.find(name) != pvIdMap.end())
        die("More than one particle vector is called %s", name.c_str());

    if (ic != nullptr)
        ic->exec(cartComm, pv.get(), domain, 0);
    else // TODO: get rid of this
        pv->domain = domain;

    auto task_checkpoint = scheduler->getTaskId("Checkpoint");
    if (checkpointEvery > 0)
    {
        info("Will save checkpoint of particle vector '%s' every %d timesteps", name.c_str(), checkpointEvery);

        auto pvPtr = pv.get();
        scheduler->addTask( task_checkpoint, [pvPtr, this] (cudaStream_t stream) {
            pvPtr->checkpoint(cartComm, restartFolder);
        }, checkpointEvery );
    }

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

    checkWallPrototypes.push_back(std::make_tuple(wall.get(), every));

    // Let the wall know the particle vector associated with it
    wall->setup(cartComm, domain, getPVbyName(wall->name));

    info("Registered wall '%s'", name.c_str());

    wallMap[name] = std::move(wall);
}

void Simulation::registerInteraction(std::shared_ptr<Interaction> interaction)
{
    std::string name = interaction->name;
    if (interactionMap.find(name) != interactionMap.end())
        die("More than one interaction is called %s", name.c_str());

    interactionMap[name] = std::move(interaction);
}

void Simulation::registerIntegrator(std::shared_ptr<Integrator> integrator)
{
    std::string name = integrator->name;
    if (integratorMap.find(name) != integratorMap.end())
        die("More than one integrator is called %s", name.c_str());

    integratorMap[name] = std::move(integrator);
}

void Simulation::registerBouncer(std::shared_ptr<Bouncer> bouncer)
{
    std::string name = bouncer->name;
    if (bouncerMap.find(name) != bouncerMap.end())
        die("More than one bouncer is called %s", name.c_str());

    bouncerMap[name] = std::move(bouncer);
}

void Simulation::registerObjectBelongingChecker(std::shared_ptr<ObjectBelongingChecker> checker)
{
    std::string name = checker->name;
    if (belongingCheckerMap.find(name) != belongingCheckerMap.end())
        die("More than one splitter is called %s", name.c_str());

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

    integrator->setPrerequisites(pv);

    integratorsStage1.push_back([integrator, pv] (float t, cudaStream_t stream) {
        integrator->stage1(pv, t, stream);
    });

    integratorsStage2.push_back([integrator, pv] (float t, cudaStream_t stream) {
        integrator->stage2(pv, t, stream);
    });
}

void Simulation::setInteraction(std::string interactionName, std::string pv1Name, std::string pv2Name)
{
    auto pv1 = getPVbyNameOrDie(pv1Name);
    auto pv2 = getPVbyNameOrDie(pv2Name);

    if (interactionMap.find(interactionName) == interactionMap.end())
        die("No such interaction: %s", interactionName.c_str());
    auto interaction = interactionMap[interactionName].get();

    interaction->setPrerequisites(pv1, pv2);

    float rc = interaction->rc;
    interactionPrototypes.push_back(std::make_tuple(rc, pv1, pv2, interaction));
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
    bouncerPrototypes.push_back(std::make_tuple(bouncer, pv));
}

void Simulation::setWallBounce(std::string wallName, std::string pvName)
{
    auto pv = getPVbyNameOrDie(pvName);

    if (wallMap.find(wallName) == wallMap.end())
        die("No such wall: %s", wallName.c_str());
    auto wall = wallMap[wallName].get();

    wall->setPrerequisites(pv);
    wallPrototypes.push_back( std::make_tuple(wall, pv) );
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
            std::string source, std::string inside, std::string outside, int checkEvery)
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
        pvInside = std::make_shared<ParticleVector>(inside, pvSource->mass);
        registerParticleVector(pvInside, nullptr, 0);
    }

    if (outside != "none" && getPVbyName(outside) == nullptr)
    {
        pvOutside = std::make_shared<ParticleVector>(outside, pvSource->mass);
        registerParticleVector(pvOutside, nullptr, 0);
    }

    splitterPrototypes.push_back(std::make_tuple(checker, pvSource, getPVbyName(inside), getPVbyName(outside)));

    belongingCorrectionPrototypes.push_back(std::make_tuple(checker, getPVbyName(inside), getPVbyName(outside), checkEvery));
}


void Simulation::prepareCellLists()
{
    info("Preparing cell-lists");

    std::map<ParticleVector*, std::vector<float>> cutOffMap;

    // Deal with the cell-lists and interactions
    for (auto prototype : interactionPrototypes)
    {
        float rc = std::get<0>(prototype);
        cutOffMap[std::get<1>(prototype)].push_back(rc);
        cutOffMap[std::get<2>(prototype)].push_back(rc);
    }

    for (auto& cutoffs : cutOffMap)
    {
        std::sort(cutoffs.second.begin(), cutoffs.second.end(), [] (float a, float b) { return a > b; });

        auto it = std::unique(cutoffs.second.begin(), cutoffs.second.end(), [=] (float a, float b) { return fabs(a - b) < rcTolerance; });
        cutoffs.second.resize( std::distance(cutoffs.second.begin(), it) );

        bool primary = true;

        // Don't use primary cell-lists with ObjectVectors
        if (dynamic_cast<ObjectVector*>(cutoffs.first) != nullptr)
            primary = false;

        for (auto rc : cutoffs.second)
        {
            cellListMap[cutoffs.first].push_back(primary ?
                    std::make_unique<PrimaryCellList>(cutoffs.first, rc, domain.localSize) :
                    std::make_unique<CellList>       (cutoffs.first, rc, domain.localSize));
            primary = false;
        }
    }
}

void Simulation::prepareInteractions()
{
    info("Preparing interactions");

    for (auto prototype : interactionPrototypes)
    {
        auto  rc = std::get<0>(prototype);
        auto pv1 = std::get<1>(prototype);
        auto pv2 = std::get<2>(prototype);

        auto& clVec1 = cellListMap[pv1];
        auto& clVec2 = cellListMap[pv2];

        CellList *cl1, *cl2;

        // Choose a CL with smallest but bigger than rc cell
        float mindiff = 10;
        for (auto& cl : clVec1)
            if (cl->rc - rc > -rcTolerance && cl->rc - rc < mindiff)
            {
                cl1 = cl.get();
                mindiff = cl->rc - rc;
            }

        mindiff = 10;
        for (auto& cl : clVec2)
            if (cl->rc - rc > -rcTolerance && cl->rc - rc < mindiff)
            {
                cl2 = cl.get();
                mindiff = cl->rc - rc;
            }

        auto inter = std::get<3>(prototype);

        regularInteractions.push_back([inter, pv1, pv2, cl1, cl2] (float t, cudaStream_t stream) {
            inter->regular(pv1, pv2, cl1, cl2, t, stream);
        });

        haloInteractions.push_back([inter, pv1, pv2, cl1, cl2] (float t, cudaStream_t stream) {
            inter->halo(pv1, pv2, cl1, cl2, t, stream);
        });
    }
}

void Simulation::prepareBouncers()
{
    info("Preparing object bouncers");

    for (auto prototype : bouncerPrototypes)
    {
        auto bouncer = std::get<0>(prototype);
        auto pv = std::get<1>(prototype);

        auto& clVec = cellListMap[pv];

        if (clVec.empty()) continue;

        CellList *cl = clVec[0].get();

        regularBouncers.push_back([bouncer, pv, cl] (float dt, cudaStream_t stream) {
            bouncer->bounceLocal(pv, cl, dt, stream);
        });

        haloBouncers.   push_back([bouncer, pv, cl] (float dt, cudaStream_t stream) {
            bouncer->bounceHalo (pv, cl, dt, stream);
        });
    }
}

void Simulation::prepareWalls()
{
    info("Preparing walls");

    for (auto prototype : wallPrototypes)
    {
        auto wall  = std::get<0>(prototype);
        auto pv    = std::get<1>(prototype);

        auto& clVec = cellListMap[pv];

        if (clVec.empty()) continue;

        CellList *cl = clVec[0].get();

        wall->attach(pv, cl);
    }

    for (auto& wall : wallMap)
    {
        auto wallPtr = wall.second.get();

        // TODO: add a property to the PVs so that they are not considered by the wall removal

        // All the particles should be removed from within the wall,
        // even those that do not interact with it
        // Only intrinsic wall particles need to remain
        for (auto& anypv : particleVectors)
            if (anypv->name != wallPtr->name)
                wallPtr->removeInner(anypv.get());
    }
}

void Simulation::execSplitters()
{
    info("Splitting particle vectors with respect to object belonging");

    for (auto prototype : splitterPrototypes)
    {
        auto checker = std::get<0>(prototype);
        auto src     = std::get<1>(prototype);
        auto inside  = std::get<2>(prototype);
        auto outside = std::get<3>(prototype);

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

    CUDA_Check( cudaDeviceSynchronize() );

    info("Preparing plugins");
    for (auto& pl : plugins)
    {
        debug("Setup and handshake of plugin %s", pl->name.c_str());
        pl->setup(this, cartComm, interComm);
        pl->handshake();
    }

    halo = std::make_unique <ParticleHaloExchanger> (cartComm, gpuAwareMPI);
    redistributor = std::make_unique <ParticleRedistributor> (cartComm, gpuAwareMPI);

    objHalo = std::make_unique <ObjectHaloExchanger> (cartComm, gpuAwareMPI);
    objRedistibutor = std::make_unique <ObjectRedistributor> (cartComm, gpuAwareMPI);
    objHaloForces = std::make_unique <ObjectForcesReverseExchanger> (cartComm, objHalo.get(), gpuAwareMPI);

    debug("Attaching particle vectors to halo exchanger and redistributor");
    for (auto& pv : particleVectors)
    {
        auto pvPtr = pv.get();

        if (cellListMap[pvPtr].size() > 0)
            if (dynamic_cast<ObjectVector*>(pvPtr) == nullptr)
            {
                auto cl = cellListMap[pvPtr][0].get();

                halo->attach         (pvPtr, cl);
                redistributor->attach(pvPtr, cl);
            }
            else
            {
                auto cl = cellListMap[pvPtr][0].get();
                auto ov = dynamic_cast<ObjectVector*>(pvPtr);

                objHalo->        attach(ov, cl->rc);
                objHaloForces->  attach(ov);
                objRedistibutor->attach(ov, cl->rc);
            }
    }

    assemble();
}

void Simulation::assemble()
{
    // XXX: different dt not implemented
    dt = 1.0;
    for (auto& integr : integratorMap)
        dt = min(dt, integr.second->dt);

    auto task_cellLists                 = scheduler->createTask("Build cell-lists");
    auto task_clearForces               = scheduler->createTask("Clear forces");
    auto task_pluginsBeforeForces       = scheduler->createTask("Plugins: before forces");
    auto task_haloInit                  = scheduler->createTask("Halo init");
    auto task_localForces               = scheduler->createTask("Local forces");
    auto task_pluginsSerializeSend      = scheduler->createTask("Plugins: serialize and send");
    auto task_haloFinalize              = scheduler->createTask("Halo finalize");
    auto task_haloForces                = scheduler->createTask("Halo forces");
    auto task_accumulateForces          = scheduler->createTask("Accumulate forces");
    auto task_pluginsBeforeIntegration  = scheduler->createTask("Plugins: before integration");
    auto task_objHaloInit               = scheduler->createTask("Object halo init");
    auto task_objHaloFinalize           = scheduler->createTask("Object halo finalize");
    auto task_clearObjHaloForces        = scheduler->createTask("Clear object halo forces");
    auto task_clearObjLocalForces       = scheduler->createTask("Clear object local forces");
    auto task_objLocalBounce            = scheduler->createTask("Local object bounce");
    auto task_objHaloBounce             = scheduler->createTask("Halo object bounce");
    auto task_correctObjBelonging       = scheduler->createTask("Correct object belonging");
    auto task_objForcesInit             = scheduler->createTask("Object forces exchange: init");
    auto task_objForcesFinalize         = scheduler->createTask("Object forces exchange: finalize");
    auto task_wallBounce                = scheduler->createTask("Wall bounce");
    auto task_wallCheck                 = scheduler->createTask("Wall check");
    auto task_pluginsAfterIntegration   = scheduler->createTask("Plugins: after integration");
    auto task_integration               = scheduler->createTask("Integration");
    auto task_redistributeInit          = scheduler->createTask("Redistribute init");
    auto task_redistributeFinalize      = scheduler->createTask("Redistribute finalize");
    auto task_objRedistInit             = scheduler->createTask("Object redistribute init");
    auto task_objRedistFinalize         = scheduler->createTask("Object redistribute finalize");


    for (auto& clVec : cellListMap)
        for (auto& cl : clVec.second)
        {
            auto clPtr = cl.get();
            scheduler->addTask(task_cellLists, [clPtr] (cudaStream_t stream) { clPtr->build(stream); } );
        }

    // Only particle forces, not object ones here
    for (auto& pv : particleVectors)
        for (auto& cl : cellListMap[pv.get()])
        {
            auto clPtr = cl.get();
            scheduler->addTask(task_clearForces, [clPtr] (cudaStream_t stream) { clPtr->forces->clear(stream); } );
        }

    for (auto& pl : plugins)
    {
        auto plPtr = pl.get();

        scheduler->addTask(task_pluginsBeforeForces, [plPtr, this] (cudaStream_t stream) {
            plPtr->setTime(currentTime, currentStep);
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
    }


    // If we have any non-object vectors
    if (particleVectors.size() != objectVectors.size())
    {
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


    for (auto& inter : regularInteractions)
        scheduler->addTask(task_localForces, [inter, this] (cudaStream_t stream) {
            inter(currentTime, stream);
        });


    for (auto& inter : haloInteractions)
        scheduler->addTask(task_haloForces, [inter, this] (cudaStream_t stream) {
            inter(currentTime, stream);
        });

    for (auto& clVec : cellListMap)
        for (auto& cl : clVec.second)
        {
            auto clPtr = cl.get();
            scheduler->addTask(task_accumulateForces, [clPtr] (cudaStream_t stream) {
                clPtr->addForces(stream);
            });
        }


    for (auto& integrator : integratorsStage2)
        scheduler->addTask(task_integration, [integrator, this] (cudaStream_t stream) {
            integrator(currentTime, stream);
        });


    for (auto ov : objectVectors)
        scheduler->addTask(task_clearObjHaloForces, [ov] (cudaStream_t stream) {
            ov->halo()->forces.clear(stream);
        });

    // As there are no primary cell-lists for objects
    // we need to separately clear real obj forces and forces in the cell-lists
    for (auto ov : objectVectors)
    {
        scheduler->addTask(task_clearObjLocalForces, [ov] (cudaStream_t stream) {
            ov->local()->forces.clear(stream);
        });

        auto& clVec = cellListMap[ov];
        for (auto& cl : clVec)
        {
            auto clPtr = cl.get();
            scheduler->addTask(task_clearObjLocalForces, [clPtr] (cudaStream_t stream) {
                clPtr->forces->clear(stream);
            });
        }
    }

    for (auto& bouncer : regularBouncers)
        scheduler->addTask(task_objLocalBounce, [bouncer, this] (cudaStream_t stream) {
            bouncer(dt, stream);
    });

    for (auto& bouncer : haloBouncers)
        scheduler->addTask(task_objHaloBounce, [bouncer, this] (cudaStream_t stream) {
            bouncer(dt, stream);
    });

    for (auto& prototype : belongingCorrectionPrototypes)
    {
        auto checker = std::get<0>(prototype);
        auto pvIn    = std::get<1>(prototype);
        auto pvOut   = std::get<2>(prototype);
        auto every   = std::get<3>(prototype);

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
            wallPtr->bounce(dt, stream);
        });
    }

    for (auto& prototype : checkWallPrototypes)
    {
        auto wall  = std::get<0>(prototype);
        auto every = std::get<1>(prototype);

        if (every > 0)
            scheduler->addTask(task_wallCheck, [this, wall] (cudaStream_t stream) { wall->check(stream); }, every);
    }


    scheduler->addDependency(scheduler->getTaskId("Checkpoint"), { task_clearForces }, { task_cellLists });

    scheduler->addDependency(task_correctObjBelonging, { task_cellLists }, {});

    scheduler->addDependency(task_cellLists, {task_clearForces}, {});

    scheduler->addDependency(task_pluginsBeforeForces, {task_localForces, task_haloForces}, {task_clearForces});
    scheduler->addDependency(task_pluginsSerializeSend, {task_redistributeInit, task_objRedistInit}, {task_pluginsBeforeForces});

    scheduler->addDependency(task_localForces, {}, {task_pluginsBeforeForces});

    scheduler->addDependency(task_clearObjHaloForces, {task_objHaloBounce}, {task_objHaloFinalize});

    scheduler->addDependency(task_objForcesInit, {}, {task_haloForces});
    scheduler->addDependency(task_objForcesFinalize, {task_accumulateForces}, {task_objForcesInit});

    scheduler->addDependency(task_haloInit, {}, {task_pluginsBeforeForces});
    scheduler->addDependency(task_haloFinalize, {}, {task_haloInit});
    scheduler->addDependency(task_haloForces, {}, {task_haloFinalize});

    scheduler->addDependency(task_accumulateForces, {task_integration}, {task_haloForces, task_localForces});
    scheduler->addDependency(task_pluginsBeforeIntegration, {task_integration}, {task_accumulateForces});
    scheduler->addDependency(task_wallBounce, {}, {task_integration});
    scheduler->addDependency(task_wallCheck, {}, {task_wallBounce});

    scheduler->addDependency(task_objHaloInit, {}, {task_integration, task_objRedistFinalize});
    scheduler->addDependency(task_objHaloFinalize, {}, {task_objHaloInit});

    scheduler->addDependency(task_objLocalBounce, {task_objHaloFinalize}, {task_integration, task_clearObjLocalForces});
    scheduler->addDependency(task_objHaloBounce, {}, {task_integration, task_objHaloFinalize, task_clearObjHaloForces});

    scheduler->addDependency(task_pluginsAfterIntegration, {task_objLocalBounce, task_objHaloBounce}, {task_integration, task_wallBounce});

    scheduler->addDependency(task_redistributeInit, {}, {task_integration, task_wallBounce, task_objLocalBounce, task_objHaloBounce, task_pluginsAfterIntegration});
    scheduler->addDependency(task_redistributeFinalize, {}, {task_redistributeInit});

    scheduler->addDependency(task_objRedistInit, {}, {task_integration, task_wallBounce, task_objForcesFinalize, task_pluginsAfterIntegration});
    scheduler->addDependency(task_objRedistFinalize, {}, {task_objRedistInit});
    scheduler->addDependency(task_clearObjLocalForces, {task_objLocalBounce}, {task_integration, task_objRedistFinalize});

    scheduler->setHighPriority(task_objForcesInit);
    scheduler->setHighPriority(task_haloInit);
    scheduler->setHighPriority(task_haloFinalize);
    scheduler->setHighPriority(task_haloForces);
    scheduler->setHighPriority(task_pluginsSerializeSend);

    scheduler->compile();

//    if (rank == 0)
//        scheduler->saveDependencyGraph_GraphML("simulation.gml");
}

void Simulation::run(int nsteps)
{
    int begin = currentStep, end = currentStep + nsteps;

    // Initial preparation
    scheduler->forceExec( scheduler->getTaskId("Object halo init"), 0 );
    scheduler->forceExec( scheduler->getTaskId("Object halo finalize"), 0 );
    scheduler->forceExec( scheduler->getTaskId("Clear object halo forces"), 0 );
    scheduler->forceExec( scheduler->getTaskId("Clear object local forces"), 0 );

    execSplitters();

    debug("Will run %d iterations now", nsteps);


    for (currentStep = begin; currentStep < end; currentStep++)
    {
        debug("===============================================================================\n"
                "Timestep: %d, simulation time: %f", currentStep, currentTime);

        scheduler->run();
        
//        MPI_Check( MPI_Barrier(cartComm) );

        currentTime += dt;
    }

    // Finish the redistribution by rebuilding the cell-lists
    scheduler->forceExec( scheduler->getTaskId("Build cell-lists"), 0 );

    info("Finished with %d iterations", nsteps);
}

void Simulation::finalize()
{
    MPI_Check( MPI_Barrier(cartComm) );

    info("Finished, exiting now");

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



