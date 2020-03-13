#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/config.h>

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
class LoaderContext;

/// A tuple that contains the Simulation and Postprocess plugins parts
using PairPlugin = std::pair<std::shared_ptr<SimulationPlugin>,
                             std::shared_ptr<PostprocessPlugin>>;

/// Logging information
struct LogInfo
{
    /// \brief Construct a LogInfo object
    LogInfo(const std::string& fileName, int verbosityLvl, bool noSplash = false);

    std::string fileName; ///< file to dump the logs to
    int verbosityLvl;     ///< higher = more debug output
    bool noSplash;        ///< if \c true, will not print the mirheo hello message
};

/** Coordinator class for a full simulation.
    Manages and splits work between \c Simulation and \c Postprocess ranks.
 */
class Mirheo
{
public:
    /** \brief Construct a \c Mirheo object using MPI_COMM_WORLD.
        \param nranks3D Number of ranks along each cartesian direction. 
        \param globalDomainSize The full domain dimensions in length units. Must be positive.
        \param dt The simulation time step
        \param logInfo Information about logging
        \param checkpointInfo Information about checkpoint
        \param gpuAwareMPI \c true to use RDMA (must be compile with a MPI version that supports it)
        \note MPI will be initialized internally. 
              If this constructor is used, the destructor will also finalize MPI.

        The product of \p nranks3D must be equal to the number of available ranks (or hals if postprocess is used)
     */
    Mirheo(int3 nranks3D, real3 globalDomainSize, real dt,
           LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI=false);

    /** \brief Construct a \c Mirheo object using a given communicator.
        \note MPI will be NOT be initialized. 
              If this constructor is used, the destructor will NOT finalize MPI.
     */
    Mirheo(MPI_Comm comm, int3 nranks3D, real3 globalDomainSize, real dt,
           LogInfo logInfo, CheckpointInfo checkpointInfo, bool gpuAwareMPI=false);

    /** \brief Construct a \c Mirheo object from a snapshot using MPI_COMM_WORLD.
        \param nranks3D Number of ranks along each cartesian direction. 
        \param snapshotPath The folder path containing the snapshot
        \param logInfo Information about logging
        \param gpuAwareMPI \c true to use RDMA (must be compile with a MPI version that supports it)
        \note MPI will be initialized internally. 
              If this constructor is used, the destructor will also finalize MPI.

        The product of \p nranks3D must be equal to the number of available ranks (or hals if postprocess is used)
     */
    Mirheo(int3 nranks3D, const std::string &snapshotPath,
           LogInfo logInfo, bool gpuAwareMPI=false);

    /** \brief Construct a \c Mirheo object from snapshot using a given communicator.
        \note MPI will be NOT be initialized. 
              If this constructor is used, the destructor will NOT finalize MPI.
     */
    Mirheo(MPI_Comm comm, int3 nranks3D, const std::string &snapshotPath,
           LogInfo logInfo, bool gpuAwareMPI=false);

    ~Mirheo();
    
    void restart(std::string folder="restart/"); ///< reset the internal state from a checkpoint folder
    bool isComputeTask() const;  ///< \return \c true if the current rank is a \c Simulation rank
    bool isMasterTask() const;   ///< \return \c true if the current rank is the root (i.e. rank = 0)
    bool isSimulationMasterTask() const;  ///< \return \c true if the current rank is the root within the simulation communicator
    bool isPostprocessMasterTask() const; ///< \return \c true if the current rank is the root within the postprocess communicator
    void startProfiler(); ///< start profiling for nvvp
    void stopProfiler();  ///< stop profiling for nvvp

    /** \brief dump the task dependency of the simulation in graphML format.
        \param fname The file name to dump the graph to (without extension).
        \param current if \c true, will only dump the current tasks; otherwise, will dump all possible ones.
    */
    void dumpDependencyGraphToGraphML(const std::string& fname, bool current) const;
    
    void run(int niters); ///< advance the system for a given number of time steps

    /** \brief register a ParticleVector in the simulation and initialize it with the gien InitialConditions.
        \param pv The ParticleVector to register
        \param ic The InitialConditions that will be applied to \p pv when registered
    */
    void registerParticleVector(const std::shared_ptr<ParticleVector>& pv, const std::shared_ptr<InitialConditions>& ic);
    
    /** \brief register an \c Interaction
        \param interaction the \c Interaction to register.
        \see setInteraction().
     */ 
    void registerInteraction(const std::shared_ptr<Interaction>& interaction);

    /** \brief register an \c Integrator
        \param integrator the \c Integrator to register.
        \see setIntegrator().
    */ 
    void registerIntegrator(const std::shared_ptr<Integrator>& integrator);

    /** \brief register a \c Wall
        \param wall The \c Wall to register
        \param checkEvery The particles that will bounce against this wall will be checked (inside/outside log info) 
               every this number of time steps. 0 means no check.
    */
    void registerWall(const std::shared_ptr<Wall>& wall, int checkEvery=0);

    /** \brief register a \c Bouncer
        \param bouncer the \c Bouncer to register.
        \see setBouncer().
    */ 
    void registerBouncer(const std::shared_ptr<Bouncer>& bouncer);

    /** \brief register a SimulationPlugin
        \param simPlugin the SimulationPlugin to register (only relevant if the current rank is a compute task).
        \param postPlugin the PostprocessPlugin to register (only relevant if the current rank is a postprocess task).
    */ 
    void registerPlugins(const std::shared_ptr<SimulationPlugin>& simPlugin,
                         const std::shared_ptr<PostprocessPlugin>& postPlugin);

    /// More generic version of registerPlugins()
    void registerPlugins(const PairPlugin &plugins);
    
    /** \brief register a ObjectBelongingChecker
        \param checker the ObjectBelongingChecker to register.
        \param ov the associated ObjectVector (must be registered).
        \see applyObjectBelongingChecker()
    */ 
    void registerObjectBelongingChecker(const std::shared_ptr<ObjectBelongingChecker>& checker, ObjectVector *ov);
 
    /** \brief Assign a registered \c Integrator to a registered ParticleVector.
        \param integrator The registered integrator (will die if it was not registered)
        \param pv The registered ParticleVector (will die if it was not registered)
     */    
    void setIntegrator(Integrator *integrator,  ParticleVector *pv);

    /** \brief Assign two registered \c Interaction to two registered ParticleVector objects.
        \param interaction The registered interaction (will die if it is not registered)
        \param pv1 The first registered ParticleVector (will die if it is not registered)
        \param pv2 The second registered ParticleVector (will die if it is not registered)

        This was designed to handle PairwiseInteraction, which needs up to two ParticleVector.
        For self interaction cases (such as MembraneInteraction), \p pv1 and \p pv2 must be the same.
    */    
    void setInteraction(Interaction *interaction, ParticleVector *pv1, ParticleVector *pv2);

    /** \brief Assign a registered \c Bouncer to registered ObjectVector and ParticleVector.
        \param bouncer The registered bouncer (will die if it is not registered)
        \param ov The registered ObjectVector that contains the surface to bounce on (will die if it is not registered)
        \param pv The registered ParticleVector to bounce (will die if it is not registered)
    */    
    void setBouncer(Bouncer *bouncer, ObjectVector *ov, ParticleVector *pv);

    /** \brief Set a registered ParticleVector to bounce on a registered \c Wall.
        \param wall The registered wall (will die if it is not registered)
        \param pv The registered ParticleVector (will die if it is not registered)
        \param maximumPartTravel Performance parameter. See \c Wall for more information.
    */    
    void setWallBounce(Wall *wall, ParticleVector *pv, real maximumPartTravel = 0.25f);

    MirState* getState(); ///< \return the global state of the system
    const MirState* getState() const; ///< \return the global state of the system (const version)
    Simulation* getSimulation();  ///< \return the Simulation object; \c nullptr on postprocess tasks.
    const Simulation* getSimulation() const;  ///< see getSimulation(); const version
    std::shared_ptr<MirState> getMirState(); ///< see getMirState(); shared_ptr version

    /** \brief Compute the SDF field from the given walls and dump it to a file in xmf+h5 format.
        \param walls List of \c Wall objects. The union of these walls will be dumped.
        \param h The grid spacing
        \param filename The base name of the dumped files (without extension)
     */ 
    void dumpWalls2XDMF(std::vector<std::shared_ptr<Wall>> walls, real3 h, const std::string& filename);

    /** \brief Compute the volume inside the geometry formed by the given walls with simple Monte-Carlo integration.
        \param walls List of \c Wall objects. The union of these walls form the geometry.
        \param nSamplesPerRank The number of Monte-Carlo samples per rank
        \return The Monte-Carlo estimate of the volume
     */
    double computeVolumeInsideWalls(std::vector<std::shared_ptr<Wall>> walls, long nSamplesPerRank = 100000);

    /** \brief Create a layer of frozen particles inside the given walls.
        \param pvName The name of the frozen ParticleVector that will be created
        \param walls The list of registered walls that need frozen particles
        \param interactions List of interactions (not necessarily registered) that will be used to equilibrate the particles
        \param integrator \c Integrator object used to equilibrate the particles
        \param numDensity The number density used to initialize the particles
        \param mass The mass of one particle
        \param nsteps Number of equilibration steps
        \return The frozen particles

        This will run a simulation of "bulk" particles and select the particles that are inside the effective 
        cut-off radius of the given list of interactions.
     */
    std::shared_ptr<ParticleVector> makeFrozenWallParticles(std::string pvName,
                                                            std::vector<std::shared_ptr<Wall>> walls,
                                                            std::vector<std::shared_ptr<Interaction>> interactions,
                                                            std::shared_ptr<Integrator> integrator,
                                                            real numDensity, real mass, int nsteps);

    /** \brief Create frozen particles inside the given objects.
        \param checker The ObjectBelongingChecker to split inside particles
        \param shape The ObjectVector that will be used to define inside particles
        \param icShape The InitialConditions object used to set the objects positions
        \param interactions List of interactions (not necessarily registered) that will be used to equilibrate the particles
        \param integrator \c Integrator object used to equilibrate the particles
        \param numDensity The number density used to initialize the particles
        \param mass The mass of one particle
        \param nsteps Number of equilibration steps
        \return The frozen particles, with name "inside_" + name of \p shape

        This will run a simulation of "bulk" particles and select the particles that are inside \p shape.
        \note For now, the output ParticleVector has mass 1.0.
     */
    std::shared_ptr<ParticleVector> makeFrozenRigidParticles(std::shared_ptr<ObjectBelongingChecker> checker,
                                                             std::shared_ptr<ObjectVector> shape,
                                                             std::shared_ptr<InitialConditions> icShape,
                                                             std::vector<std::shared_ptr<Interaction>> interactions,
                                                             std::shared_ptr<Integrator>   integrator,
                                                             real numDensity, real mass, int nsteps);
    
    /** \brief Enable a registered ObjectBelongingChecker to split particles of a registered ParticleVector.
        \param checker The ObjectBelongingChecker (will die if it is not registered)
        \param pv The registered ParticleVector that must be split (will die if it is not registered)
        \param checkEvery The particle split will be performed every this amount of time steps.
        \param inside Name of the ParticleVector that will contain the particles of \p pv that are inside the objects. See below for more information.
        \param outside Name of the ParticleVector that will contain the particles of \p pv that are outside the objects. See below for more information.

        \p inside or \p outside can take the reserved value "none", in which case the corresponding particles will be deleted.
        Furthermore, exactly one of \p inside and \p outside must be the same as \p pv.
        
        If \p inside or \p outside has the name of a ParticleVector that is not registered, this call will create an empty ParticleVector 
        with the given name  and register it in the \c Simulation.
        Otherwise the already registered ParticleVector will be used.
     */
    std::shared_ptr<ParticleVector> applyObjectBelongingChecker(ObjectBelongingChecker *checker,
                                                                ParticleVector *pv,
                                                                int checkEvery,
                                                                std::string inside = "",
                                                                std::string outside = "");    

    /// print the list of all compile options and their current value in the logs
    void logCompileOptions() const;

    /** \brief Save snapshot of the Mirheo simulation to the given folder.
        \param [in] path The target folder path.
      */
    void saveSnapshot(std::string path);

    /** \brief Set a user-defined attribute to the given value. Useful for attaching extra information to snapshot.
        \param [in] name The attribute name.
        \param [in] value The attribute value. Can be an integer, floating point number, array or an object (dictionary).
      */
    void setAttribute(const std::string& name, ConfigValue value);

    /** \brief Read a user-defined attribute of the given name as an integer.
        \param [in] name The attribute name.
        \return The attribute value. Throws an exception if the attribute is not found or the value is not an integer.
      */
    const ConfigValue& getAttribute(const std::string& name);

private:
    std::unique_ptr<Simulation> sim_;
    std::unique_ptr<Postprocess> post_;
    std::shared_ptr<MirState> state_;
    ConfigObject attributes_;
    
    int rank_;
    int computeTask_;
    bool noPostprocess_;
    int pluginsTag_ {0}; ///< used to create unique tag per plugin
    
    bool initialized_    = false;
    bool initializedMpi_ = false;

    MPI_Comm comm_      {MPI_COMM_NULL}; ///< base communicator (world)
    MPI_Comm cartComm_  {MPI_COMM_NULL}; ///< Cartesian communicator for simulation part; might be from comm if no postprocess
    MPI_Comm ioComm_    {MPI_COMM_NULL}; ///< postprocess communicator
    MPI_Comm compComm_  {MPI_COMM_NULL}; ///< simulation communicator
    MPI_Comm interComm_ {MPI_COMM_NULL}; ///< intercommunicator between postprocess and simulation

    void init(int3 nranks3D, real3 globalDomainSize, real dt, LogInfo logInfo,
              CheckpointInfo checkpointInfo, bool gpuAwareMPI,
              LoaderContext *context = nullptr);
    void initFromSnapshot(int3 nranks3D, const std::string &snapshotPath,
                          LogInfo logInfo, bool gpuAwareMPI);
    void initLogger(MPI_Comm comm, LogInfo logInfo);
    void sayHello();
    void setup();
    void ensureNotInitialized() const;
};

} // namespace mirheo
