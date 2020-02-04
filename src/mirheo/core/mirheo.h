#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/common.h>

#include <memory>
#include <mpi.h>
#include <vector_types.h>
#include <vector>

namespace mirheo
{

class MirState;

class Simulation;
class Postprocess;

class ParticleVector;
class ObjectVector;
class InitialConditions;
class Integrator;
class Interaction;
class ObjectBelongingChecker;
class Bouncer;
class Wall;
class SimulationPlugin;
class PostprocessPlugin;

using PairPlugin = std::pair<std::shared_ptr<SimulationPlugin>,
                             std::shared_ptr<PostprocessPlugin>>;

struct LogInfo
{
    LogInfo(const std::string& fileName, int verbosityLvl, bool noSplash = false);
    std::string fileName;
    int verbosityLvl;
    bool noSplash;
};

class Mirheo
{
public:
    Mirheo(int3 nranks3D, real3 globalDomainSize, real dt,
           LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI=false);

    Mirheo(long commAddress, int3 nranks3D, real3 globalDomainSize, real dt,
           LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI=false);

    Mirheo(MPI_Comm comm, int3 nranks3D, real3 globalDomainSize, real dt,
           LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI=false);

    ~Mirheo();
    
    void restart(std::string folder="restart/");
    bool isComputeTask() const;
    bool isMasterTask() const;
    void startProfiler();
    void stopProfiler();
    void saveDependencyGraph_GraphML(std::string fname, bool current) const;
    
    void run(int niters);
    
    void registerParticleVector         (const std::shared_ptr<ParticleVector>& pv,
                                         const std::shared_ptr<InitialConditions>& ic);
    
    void registerInteraction            (const std::shared_ptr<Interaction>& interaction);
    void registerIntegrator             (const std::shared_ptr<Integrator>& integrator);
    void registerWall                   (const std::shared_ptr<Wall>& wall, int checkEvery=0);
    void registerBouncer                (const std::shared_ptr<Bouncer>& bouncer);
    void registerPlugins                (const std::shared_ptr<SimulationPlugin>& simPlugin,
                                         const std::shared_ptr<PostprocessPlugin>& postPlugin);
    void registerPlugins                (const PairPlugin &plugins);
    
    void registerObjectBelongingChecker (const std::shared_ptr<ObjectBelongingChecker>& checker, ObjectVector* ov);
 
    void setIntegrator  (Integrator *integrator,  ParticleVector *pv);
    void setInteraction (Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2);
    void setBouncer     (Bouncer *bouncer, ObjectVector *ov, ParticleVector *pv);
    void setWallBounce  (Wall *wall, ParticleVector *pv, real maximumPartTravel = 0.25f);

    MirState* getState();
    const MirState* getState() const;
    Simulation* getSimulation();
    const Simulation* getSimulation() const;
    std::shared_ptr<MirState> getMirState();

    void dumpWalls2XDMF(std::vector<std::shared_ptr<Wall>> walls, real3 h, const std::string& filename);
    double computeVolumeInsideWalls(std::vector<std::shared_ptr<Wall>> walls, long nSamplesPerRank = 100000);
    
    std::shared_ptr<ParticleVector> makeFrozenWallParticles(std::string pvName,
                                                            std::vector<std::shared_ptr<Wall>> walls,
                                                            std::vector<std::shared_ptr<Interaction>> interactions,
                                                            std::shared_ptr<Integrator> integrator,
                                                            real density, int nsteps);

    std::shared_ptr<ParticleVector> makeFrozenRigidParticles(std::shared_ptr<ObjectBelongingChecker> checker,
                                                             std::shared_ptr<ObjectVector> shape,
                                                             std::shared_ptr<InitialConditions> icShape,
                                                             std::vector<std::shared_ptr<Interaction>> interactions,
                                                             std::shared_ptr<Integrator>   integrator,
                                                             real density, int nsteps);
    
    std::shared_ptr<ParticleVector> applyObjectBelongingChecker(ObjectBelongingChecker* checker,
                                                                ParticleVector* pv,
                                                                int checkEvery,
                                                                std::string inside = "",
                                                                std::string outside = "");    

    void logCompileOptions() const;
    Config getConfig() const;
    void storeConfigJSON(const std::string &computeFilename,
                         const std::string &postprocessFilename) const;
    
private:
    std::unique_ptr<Simulation> sim_;
    std::unique_ptr<Postprocess> post_;
    std::shared_ptr<MirState> state_;
    
    int rank_;
    int computeTask_;
    bool noPostprocess_;
    int pluginsTag_ {0}; ///< used to create unique tag per plugin
    
    bool initialized_    = false;
    bool initializedMpi_ = false;

    MPI_Comm comm_      {MPI_COMM_NULL}; ///< base communicator (world)
    MPI_Comm cartComm_  {MPI_COMM_NULL}; ///< cartesian communicator for simulation part; might be from comm if no postprocess
    MPI_Comm ioComm_    {MPI_COMM_NULL}; ///< postprocess communicator
    MPI_Comm compComm_  {MPI_COMM_NULL}; ///< simulation communicator
    MPI_Comm interComm_ {MPI_COMM_NULL}; ///< intercommunicator between postprocess and simulation

    void init(int3 nranks3D, real3 globalDomainSize, real dt, LogInfo logInfo,
              CheckpointInfo checkpointInfo, bool gpuAwareMPI);
    void initLogger(MPI_Comm comm, LogInfo logInfo);
    void sayHello();
    void setup();
    void ensureNotInitialized() const;
};

} // namespace mirheo
