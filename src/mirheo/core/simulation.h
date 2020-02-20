#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/exchangers/exchanger_interfaces.h>
#include <mirheo/core/mirheo_object.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mirheo
{

class Saver;
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

    /** \brief Dump all simulation data, create a ConfigObject describing the simulation state and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly \c Simulation.
      */
    void saveSnapshotAndRegister(Saver& saver) override;

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
    ObjectVector*   getOVbyName     (const std::string& name) const;
    ObjectVector*   getOVbyNameOrDie(const std::string& name) const;
    
    /// Assume co-ownership
    std::shared_ptr<ParticleVector> getSharedPVbyName(const std::string& name) const;

    Wall* getWallByNameOrDie(const std::string& name) const;

    CellList* gelCellList(ParticleVector* pv) const;

    void startProfiler() const;
    void stopProfiler() const;

    MPI_Comm getCartComm() const;
    int3 getRank3D() const;
    int3 getNRanks3D() const;
    
    real getCurrentDt() const;
    real getCurrentTime() const;

    real getMaxEffectiveCutoff() const;
    
    void saveDependencyGraph_GraphML(const std::string& fname, bool current) const;

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

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

private:
    friend Saver;

    const int3 nranks3D_;
    const int3 rank3D_;

    MPI_Comm cartComm_;
    MPI_Comm interComm_;
    
    MirState *state_;
    
    static constexpr real rcTolerance_ = 1e-5_r;

    int checkpointId_ {0};
    const CheckpointInfo checkpointInfo_;
    const int rank_;

    std::unique_ptr<TaskScheduler> scheduler_;
    std::unique_ptr<SimulationTasks> tasks_;

    std::unique_ptr<InteractionManager> interactionsIntermediate_, interactionsFinal_;

    const bool gpuAwareMPI_;

    ExchangeEngineUniquePtr partRedistributor_, objRedistibutor_;
    ExchangeEngineUniquePtr partHaloIntermediate_, partHaloFinal_;
    ExchangeEngineUniquePtr objHaloIntermediate_, objHaloReverseIntermediate_;
    ExchangeEngineUniquePtr objHaloFinal_, objHaloReverseFinal_;

    std::map<std::string, int> pvIdMap_;
    std::vector< std::shared_ptr<ParticleVector> > particleVectors_;
    std::vector< ObjectVector* >   objectVectors_;

    MapShared <Bouncer>                bouncerMap_;
    MapShared <Integrator>             integratorMap_;
    MapShared <Interaction>            interactionMap_;
    MapShared <Wall>                   wallMap_;
    MapShared <ObjectBelongingChecker> belongingCheckerMap_;
    
    std::vector< std::shared_ptr<SimulationPlugin> > plugins;

    std::map<ParticleVector*, std::vector< std::unique_ptr<CellList> >> cellListMap_;

    std::vector<IntegratorPrototype>          integratorPrototypes_;
    std::vector<InteractionPrototype>         interactionPrototypes_;
    std::vector<WallPrototype>                wallPrototypes_;
    std::vector<CheckWallPrototype>           checkWallPrototypes_;
    std::vector<BouncerPrototype>             bouncerPrototypes_;
    std::vector<BelongingCorrectionPrototype> belongingCorrectionPrototypes_;
    std::vector<SplitterPrototype>            splitterPrototypes_;

    std::vector<std::function<void(cudaStream_t)>> regularBouncers_, haloBouncers_;

    std::map<std::string, std::string> pvsIntegratorMap_;
};

} // namespace mirheo
