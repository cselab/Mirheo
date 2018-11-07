#pragma once

#include <core/logger.h>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/mpi/exchanger_interfaces.h>

#include "domain.h"

#include <tuple>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <memory>

// Some forward declarations
class ParticleVector;
class ObjectVector;
class CellList;
class TaskScheduler;

class Wall;
class Interaction;
class Integrator;
class InitialConditions;
class Bouncer;
class ObjectBelongingChecker;
class SimulationPlugin;


class Simulation
{
public:
    int3 nranks3D;
    int3 rank3D;

    MPI_Comm cartComm;
    MPI_Comm interComm;

    DomainInfo domain;

    Simulation(int3 nranks3D, float3 globalDomainSize,
               const MPI_Comm& comm, const MPI_Comm& interComm,
               int globalCheckpointEvery = 0,
               std::string restartFolder = "restart/", bool gpuAwareMPI = false);
    ~Simulation();
    
    void restart();

    void registerParticleVector         (std::shared_ptr<ParticleVector> pv, std::shared_ptr<InitialConditions> ic, int checkpointEvery=0);
    void registerWall                   (std::shared_ptr<Wall> wall, int checkEvery=0);
    void registerInteraction            (std::shared_ptr<Interaction> interaction);
    void registerIntegrator             (std::shared_ptr<Integrator> integrator);
    void registerBouncer                (std::shared_ptr<Bouncer> bouncer);
    void registerPlugin                 (std::shared_ptr<SimulationPlugin> plugin);
    void registerObjectBelongingChecker (std::shared_ptr<ObjectBelongingChecker> checker);


    void setIntegrator             (std::string integratorName,  std::string pvName);
    void setInteraction            (std::string interactionName, std::string pv1Name, std::string pv2Name);
    void setBouncer                (std::string bouncerName,     std::string objName, std::string pvName);
    void setWallBounce             (std::string wallName,        std::string pvName);
    void setObjectBelongingChecker (std::string checkerName,     std::string objName);


    void applyObjectBelongingChecker(std::string checkerName,
            std::string source, std::string inside, std::string outside, int checkEvery);


    void init();
    void run(int nsteps);

    std::vector<ParticleVector*> getParticleVectors() const;

    ParticleVector* getPVbyName     (std::string name) const;
    ParticleVector* getPVbyNameOrDie(std::string name) const;
    ObjectVector*   getOVbyNameOrDie(std::string name) const;
    
    /// Assume co-ownership
    std::shared_ptr<ParticleVector> getSharedPVbyName(std::string name) const;

    Wall* getWallByNameOrDie(std::string name) const;

    CellList* gelCellList(ParticleVector* pv) const;

    void startProfiler() const;
    void stopProfiler() const;

    MPI_Comm getCartComm() const;
    
    float getCurrentDt() const;
    
    void saveDependencyGraph_GraphML(std::string fname) const;


private:    
    const float rcTolerance = 1e-5;

    std::string restartFolder;
    int globalCheckpointEvery;

    float dt;
    int rank;

    double currentTime;
    int currentStep;

    std::unique_ptr<TaskScheduler> scheduler;

    bool gpuAwareMPI;
    std::unique_ptr<ExchangeEngine> halo;
    std::unique_ptr<ExchangeEngine> redistributor;

    std::unique_ptr<ExchangeEngine> objHalo;
    std::unique_ptr<ExchangeEngine> objRedistibutor;
    std::unique_ptr<ExchangeEngine> objHaloForces;

    std::map<std::string, int> pvIdMap;
    std::vector< std::shared_ptr<ParticleVector> > particleVectors;
    std::vector< ObjectVector* >   objectVectors;

    template <class T>
    using MapShared = std::map< std::string, std::shared_ptr<T> >;

    MapShared <Bouncer>                bouncerMap;
    MapShared <Integrator>             integratorMap;
    MapShared <Interaction>            interactionMap;
    MapShared <Wall>                   wallMap;
    MapShared <ObjectBelongingChecker> belongingCheckerMap;
    
    std::vector< std::shared_ptr<SimulationPlugin> > plugins;


    std::map<ParticleVector*, std::vector< std::unique_ptr<CellList> >> cellListMap;

    struct InteractionPrototype
    {
        float rc;
        ParticleVector *pv1, *pv2;
        Interaction *interaction;
    };

    struct WallPrototype
    {
        Wall *wall;
        ParticleVector *pv;
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
    
    std::vector<InteractionPrototype>         interactionPrototypes;
    std::vector<WallPrototype>                wallPrototypes;
    std::vector<CheckWallPrototype>           checkWallPrototypes;
    std::vector<BouncerPrototype>             bouncerPrototypes;
    std::vector<BelongingCorrectionPrototype> belongingCorrectionPrototypes;
    std::vector<SplitterPrototype>            splitterPrototypes;


    std::vector<std::function<void(float, cudaStream_t)>> regularInteractions, haloInteractions;
    std::vector<std::function<void(float, cudaStream_t)>> integratorsStage1, integratorsStage2;
    std::vector<std::function<void(float, cudaStream_t)>> regularBouncers, haloBouncers;

    
    void prepareCellLists();
    void prepareInteractions();
    void prepareBouncers();
    void prepareWalls();
    void execSplitters();
    
    void checkpoint();

    void assemble();
};






