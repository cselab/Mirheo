#pragma once

#include <core/logger.h>
#include <core/utils/common.h>
#include <core/utils/pytypes.h>

#include <memory>
#include <mpi.h>

class YmrState;

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

class YMeRo
{
public:
    YMeRo(PyTypes::int3 nranks3D, PyTypes::float3 globalDomainSize, float dt,
          std::string logFileName, int verbosity, int checkpointEvery=0,
          std::string checkpointFolder="restart/",
          CheckpointIdAdvanceMode checkpointMode = CheckpointIdAdvanceMode::PingPong,
          bool gpuAwareMPI=false, bool noSplash=false);

    YMeRo(long commAddress, PyTypes::int3 nranks3D, PyTypes::float3 globalDomainSize, float dt,
          std::string logFileName, int verbosity, int checkpointEvery=0,
          std::string checkpointFolder="restart/",
          CheckpointIdAdvanceMode checkpointMode = CheckpointIdAdvanceMode::PingPong,
          bool gpuAwareMPI=false, bool noSplash=false);

    YMeRo(MPI_Comm comm, PyTypes::int3 nranks3D, PyTypes::float3 globalDomainSize, float dt,
          std::string logFileName, int verbosity, int checkpointEvery=0,
          std::string checkpointFolder="restart/",
          CheckpointIdAdvanceMode checkpointMode = CheckpointIdAdvanceMode::PingPong,
          bool gpuAwareMPI=false, bool noSplash=false);

    ~YMeRo();
    
    void restart(std::string folder="restart/");
    bool isComputeTask() const;
    bool isMasterTask() const;
    void startProfiler();
    void stopProfiler();
    void saveDependencyGraph_GraphML(std::string fname, bool current) const;
    
    void run(int niters);
    
    void registerParticleVector         (const std::shared_ptr<ParticleVector>& pv,
                                         const std::shared_ptr<InitialConditions>& ic, int checkpointEvery);
    
    void registerInteraction            (const std::shared_ptr<Interaction>& interaction);
    void registerIntegrator             (const std::shared_ptr<Integrator>& integrator);
    void registerWall                   (const std::shared_ptr<Wall>& wall, int checkEvery=0);
    void registerBouncer                (const std::shared_ptr<Bouncer>& bouncer);
    void registerPlugins                (const std::shared_ptr<SimulationPlugin>& simPlugin,
                                         const std::shared_ptr<PostprocessPlugin>& postPlugin);
    
    void registerObjectBelongingChecker (const std::shared_ptr<ObjectBelongingChecker>& checker, ObjectVector* ov);
 
    void setIntegrator  (Integrator* integrator,  ParticleVector* pv);
    void setInteraction (Interaction* interaction, ParticleVector* pv1, ParticleVector* pv2);
    void setBouncer     (Bouncer* bouncer, ObjectVector* ov, ParticleVector* pv);
    void setWallBounce  (Wall* wall, ParticleVector* pv);

    YmrState* getState();
    const YmrState* getState() const;
    std::shared_ptr<YmrState> getYmrState();

    void dumpWalls2XDMF(std::vector<std::shared_ptr<Wall>> walls, PyTypes::float3 h, std::string filename);
    double computeVolumeInsideWalls(std::vector<std::shared_ptr<Wall>> walls, long nSamplesPerRank = 100000);
    
    std::shared_ptr<ParticleVector> makeFrozenWallParticles(std::string pvName,
                                                            std::vector<std::shared_ptr<Wall>> walls,
                                                            std::vector<std::shared_ptr<Interaction>> interactions,
                                                            std::shared_ptr<Integrator> integrator,
                                                            float density, int nsteps);

    std::shared_ptr<ParticleVector> makeFrozenRigidParticles(std::shared_ptr<ObjectBelongingChecker> checker,
                                                             std::shared_ptr<ObjectVector> shape,
                                                             std::shared_ptr<InitialConditions> icShape,
                                                             std::vector<std::shared_ptr<Interaction>> interactions,
                                                             std::shared_ptr<Integrator>   integrator,
                                                             float density, int nsteps);
    
    std::shared_ptr<ParticleVector> applyObjectBelongingChecker(ObjectBelongingChecker* checker,
                                                                ParticleVector* pv,
                                                                int checkEvery,
                                                                std::string inside = "",
                                                                std::string outside = "",
                                                                int checkpointEvery=0);    
    
private:
    std::unique_ptr<Simulation> sim;
    std::unique_ptr<Postprocess> post;
    std::shared_ptr<YmrState> state;
    
    int rank;
    int computeTask;
    bool noPostprocess;
    bool noSplash;
    
    bool initialized = false;
    bool initializedMpi = false;

    MPI_Comm comm      {MPI_COMM_NULL}; ///< base communicator (world)
    MPI_Comm cartComm  {MPI_COMM_NULL}; ///< cartesian communicator for simulation part; might be from comm if no postprocess
    MPI_Comm ioComm    {MPI_COMM_NULL}; ///< postprocess communicator
    MPI_Comm compComm  {MPI_COMM_NULL}; ///< simulation communicator
    MPI_Comm interComm {MPI_COMM_NULL}; ///< intercommunicator between postprocess and simulation

    void init(int3 nranks3D, float3 globalDomainSize, float dt, std::string logFileName, int verbosity,
              int checkpointEvery, std::string checkpointFolder, CheckpointIdAdvanceMode checkpointMode, bool gpuAwareMPI);
    void initLogger(MPI_Comm comm, std::string logFileName, int verbosity);
    void sayHello();
};
