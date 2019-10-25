#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/domain.h>
#include <core/logger.h>
#include <core/exchangers/exchanger_interfaces.h>
#include <core/mirheo_object.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

class MirState;
class ParticleVector;
class ObjectVector;
class CellList;
class TaskScheduler;
class InteractionManager;

class Wall;
class Interaction;
class Integrator;
class InitialConditions;
class Bouncer;
class ObjectBelongingChecker;
class SimulationPlugin;
struct SimulationTasks;

class Simulation : protected MirObject
{
public:

    Simulation(const MPI_Comm &cartComm, const MPI_Comm &interComm, MirState *state,
               CheckpointInfo checkpointInfo = {}, bool gpuAwareMPI = false);

    ~Simulation();
    
    void restart(const std::string& folder);
    void checkpoint();

    void registerParticleVector         (std::shared_ptr<ParticleVector> pv, std::shared_ptr<InitialConditions> ic);
    void registerWall                   (std::shared_ptr<Wall> wall, int checkEvery=0);
    void registerInteraction            (std::shared_ptr<Interaction> interaction);
    void registerIntegrator             (std::shared_ptr<Integrator> integrator);
    void registerBouncer                (std::shared_ptr<Bouncer> bouncer);
    void registerPlugin                 (std::shared_ptr<SimulationPlugin> plugin, int tag);
    void registerObjectBelongingChecker (std::shared_ptr<ObjectBelongingChecker> checker);


    void setIntegrator             (const std::string& integratorName,  const std::string& pvName);
    void setInteraction            (const std::string& interactionName, const std::string& pv1Name, const std::string& pv2Name);
    void setBouncer                (const std::string& bouncerName,     const std::string& objName, const std::string& pvName);
    void setWallBounce             (const std::string& wallName,        const std::string& pvName, real maximumPartTravel);
    void setObjectBelongingChecker (const std::string& checkerName,     const std::string& objName);


    void applyObjectBelongingChecker(const std::string& checkerName,
                                     const std::string& source, const std::string& inside, const std::string& outside,
                                     int checkEvery);


    void init();
    void run(int nsteps);

    void notifyPostProcess(int tag, int msg) const;

    std::vector<ParticleVector*> getParticleVectors() const;

    ParticleVector* getPVbyName     (const std::string& name) const;
    ParticleVector* getPVbyNameOrDie(const std::string& name) const;
    ObjectVector*   getOVbyNameOrDie(const std::string& name) const;
    
    /// Assume co-ownership
    std::shared_ptr<ParticleVector> getSharedPVbyName(const std::string& name) const;

    Wall* getWallByNameOrDie(const std::string& name) const;

    CellList* gelCellList(ParticleVector* pv) const;

    void startProfiler() const;
    void stopProfiler() const;

    MPI_Comm getCartComm() const;
    
    real getCurrentDt() const;
    real getCurrentTime() const;

    real getMaxEffectiveCutoff() const;
    
    void saveDependencyGraph_GraphML(const std::string& fname, bool current) const;

public:
    const int3 nranks3D;
    const int3 rank3D;

    MPI_Comm cartComm;
    MPI_Comm interComm;

private:

    using ExchangeEngineUniquePtr = std::unique_ptr<ExchangeEngine>;

    template <class T>
    using MapShared = std::map< std::string, std::shared_ptr<T> >;

    struct IntegratorPrototype
    {
        ParticleVector *pv;
        Integrator *integrator;
    };

    struct InteractionPrototype
    {
        real rc;
        ParticleVector *pv1, *pv2;
        Interaction *interaction;
    };

    struct WallPrototype
    {
        Wall *wall;
        ParticleVector *pv;
        real maximumPartTravel;
    };

    struct CheckWallPrototype
    {
        Wall *wall;
        int every;
    };

    struct BouncerPrototype
    {
        Bouncer *bouncer;
        ParticleVector *pv;
    };

    struct BelongingCorrectionPrototype
    {
        ObjectBelongingChecker *checker;
        ParticleVector *pvIn, *pvOut;
        int every;
    };

    struct SplitterPrototype
    {
        ObjectBelongingChecker *checker;
        ParticleVector *pvSrc, *pvIn, *pvOut;
    };

    
    MirState *state;
    
    static constexpr real rcTolerance = 1e-5;

    int checkpointId {0};
    const CheckpointInfo checkpointInfo;
    const int rank;

    std::unique_ptr<TaskScheduler> scheduler;
    std::unique_ptr<SimulationTasks> tasks;

    std::unique_ptr<InteractionManager> interactionsIntermediate, interactionsFinal;

    const bool gpuAwareMPI;

    ExchangeEngineUniquePtr partRedistributor, objRedistibutor;
    ExchangeEngineUniquePtr partHaloIntermediate, partHaloFinal;
    ExchangeEngineUniquePtr objHaloIntermediate, objHaloReverseIntermediate;
    ExchangeEngineUniquePtr objHaloFinal, objHaloReverseFinal;

    std::map<std::string, int> pvIdMap;
    std::vector< std::shared_ptr<ParticleVector> > particleVectors;
    std::vector< ObjectVector* >   objectVectors;

    MapShared <Bouncer>                bouncerMap;
    MapShared <Integrator>             integratorMap;
    MapShared <Interaction>            interactionMap;
    MapShared <Wall>                   wallMap;
    MapShared <ObjectBelongingChecker> belongingCheckerMap;
    
    std::vector< std::shared_ptr<SimulationPlugin> > plugins;

    std::map<ParticleVector*, std::vector< std::unique_ptr<CellList> >> cellListMap;

    std::vector<IntegratorPrototype>          integratorPrototypes;
    std::vector<InteractionPrototype>         interactionPrototypes;
    std::vector<WallPrototype>                wallPrototypes;
    std::vector<CheckWallPrototype>           checkWallPrototypes;
    std::vector<BouncerPrototype>             bouncerPrototypes;
    std::vector<BelongingCorrectionPrototype> belongingCorrectionPrototypes;
    std::vector<SplitterPrototype>            splitterPrototypes;

    std::vector<std::function<void(cudaStream_t)>> regularBouncers, haloBouncers;

    std::map<std::string, std::string> pvsIntegratorMap;

    
private:

    std::vector<std::string> getExtraDataToExchange(ObjectVector *ov);
    std::vector<std::string> getDataToSendBack(const std::vector<std::string>& extraOut, ObjectVector *ov);
    
    void prepareCellLists();
    void prepareInteractions();
    void prepareBouncers();
    void prepareWalls();
    void preparePlugins();
    void prepareEngines();
    
    void execSplitters();

    void createTasks();

    using MirObject::restart;
    using MirObject::checkpoint;

    void restartState(const std::string& folder);
    void checkpointState();
};

