// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/exchangers/interface.h>
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

class Wall;
class Interaction;
class Integrator;
class InitialConditions;
class Bouncer;
class ObjectBelongingChecker;
class SimulationPlugin;
struct SimulationTasks;
struct RunData;

/** \brief Manage and combine all MirObject objects to run a simulation.

    All \c MirObject objects must be registered and set before calling run().

    This must be instantiated only by ranks that have access to a GPU.
    Optionally, this class can communicate with a \c Postprocess one held on a different rank.
    This option is used for \c Plugins.
 */
class Simulation : protected MirObject
{
public:

    /** \brief Construct an empty \c Simulation object
        \param cartComm a cartesian communicator that holds all ranks of the simulation.
        \param interComm An inter communicator to communicate with the \c Postprocess ranks.
        \param [in,out] state The global state of the simulation. Does not pass ownership.
        \param checkpointInfo Configuration of checkpoint
        \param gpuAwareMPI Performance parameter that controls if communication can be performed through RDMA.
     */
    Simulation(const MPI_Comm &cartComm, const MPI_Comm &interComm, MirState *state,
               CheckpointInfo checkpointInfo, bool gpuAwareMPI = false);

    ~Simulation();

    /// restore the simulation state from a folder that contains all restart files
    void restart(const std::string& folder);

    /// Dump the whole simulation state to the checkpoint folder and advance the checkpoint ID.
    void checkpoint();

    /** \brief Dump the whole simulation state and setup, and advance the checkpoint ID.

        Target path is automatically determined from the checkpoint folder and ID.
        Wrapper for `snaphot(const std::string&)`.
      */
    void snapshot();

    /** \brief Dump the whole simulation setup and data at the given path.
        \param path Target folder.
      */
    void snapshot(const std::string& path);

    /** \brief register a ParticleVector and initialize it with the gien InitialConditions.
        \param pv The ParticleVector to register
        \param ic The InitialConditions that will be applied to \p pv when registered
     */
    void registerParticleVector(std::shared_ptr<ParticleVector> pv, std::shared_ptr<InitialConditions> ic);

    /** \brief register a \c Wall
        \param wall The \c Wall to register
        \param checkEvery The particles that will bounce against this wall will be checked (inside/outside log info)
               every this number of time steps. 0 means no check.
     */
    void registerWall(std::shared_ptr<Wall> wall, int checkEvery = 0);

    /** \brief register an \c Interaction
        \param interaction the \c Interaction to register.
        \see setInteraction().
     */
    void registerInteraction(std::shared_ptr<Interaction> interaction);

    /** \brief register an \c Integrator
        \param integrator the \c Integrator to register.
        \see setIntegrator().
    */
    void registerIntegrator(std::shared_ptr<Integrator> integrator);

    /** \brief register a \c Bouncer
        \param bouncer the \c Bouncer to register.
        \see setBouncer().
    */
    void registerBouncer(std::shared_ptr<Bouncer> bouncer);

    /** \brief register a SimulationPlugin
        \param plugin the SimulationPlugin to register.
        \param tag A unique tag per plugin, used by MPI communications. Must be different for every plugin.
        \note If there is a \c Postprocess rank, it might need to register the corrsponding PostprocessPlugin.
    */
    void registerPlugin(std::shared_ptr<SimulationPlugin> plugin, int tag);

    /** \brief register a ObjectBelongingChecker
        \param checker the ObjectBelongingChecker to register.
        \see applyObjectBelongingChecker()
    */
    void registerObjectBelongingChecker(std::shared_ptr<ObjectBelongingChecker> checker);


    /** \brief Assign a registered \c Integrator to a registered ParticleVector.
        \param integratorName Name of the registered integrator (will die if it does not exist)
        \param pvName Name of the registered ParticleVector (will die if it does not exist)
     */
    void setIntegrator(const std::string& integratorName, const std::string& pvName);

    /** \brief Assign two registered \c Interaction to two registered ParticleVector objects.
        \param interactionName Name of the registered interaction (will die if it does not exist)
        \param pv1Name Name of the first registered ParticleVector (will die if it does not exist)
        \param pv2Name Name of the second registered ParticleVector (will die if it does not exist)

        This was designed to handle PairwiseInteraction, which needs up to two ParticleVector.
        For self interaction cases (such as MembraneInteraction), \p pv1Name and \p pv2Name must be the same.
     */
    void setInteraction(const std::string& interactionName, const std::string& pv1Name, const std::string& pv2Name);

    /** \brief Assign a registered \c Bouncer to registered ObjectVector and ParticleVector.
        \param bouncerName Name of the registered bouncer (will die if it does not exist)
        \param objName Name of the registered ObjectVector that contains the surface to bounce on (will die if it does not exist)
        \param pvName Name of the registered ParticleVector to bounce (will die if it does not exist)
     */
    void setBouncer(const std::string& bouncerName, const std::string& objName, const std::string& pvName);

    /** \brief Set a registered ParticleVector to bounce on a registered \c Wall.
        \param wallName Name of the registered wall (will die if it does not exist)
        \param pvName Name of the registered ParticleVector (will die if it does not exist)
        \param maximumPartTravel Performance parameter. See \c Wall for more information.
    */
    void setWallBounce(const std::string& wallName, const std::string& pvName, real maximumPartTravel);

    /** \brief Associate a registered ObjectBelongingChecker to a registered ObjectVector.
        \param checkerName Name of the registered ObjectBelongingChecker (will die if it does not exist)
        \param objName Name of the registered ObjectVector (will die if it does not exist)
        \note this is required before calling applyObjectBelongingChecker()
     */
    void setObjectBelongingChecker(const std::string& checkerName, const std::string& objName);


    /** \brief Enable a registered ObjectBelongingChecker to split particles of a registered ParticleVector.
        \param checkerName The name of the ObjectBelongingChecker. Must be associated to an ObjectVector with setObjectBelongingChecker() (will die if it does not exist)
        \param source The registered ParticleVector that must be split (will die if it does not exist)
        \param inside Name of the ParticleVector that will contain the particles of \p source that are inside the objects. See below for more information.
        \param outside Name of the ParticleVector that will contain the particles of \p source that are outside the objects. See below for more information.
        \param checkEvery The particle split will be performed every this amount of time steps.

        \p inside or \p outside can take the reserved value "none", in which case the corresponding particles will be deleted.
        Furthermore, exactly one of \p inside and \p outside must be the same as \p source.

        If \p inside or \p outside has the name of a ParticleVector that is not registered, this call will create an empty ParticleVector
        with the given name  and register it in the \c Simulation.
        Otherwise the already registered ParticleVector will be used.
     */
    void applyObjectBelongingChecker(const std::string& checkerName, const std::string& source,
                                     const std::string& inside, const std::string& outside, int checkEvery);


    void init(); ///< setup all the simulation tasks from the registered objects and their relation. Must be called after all the register and set methods.
    void run(int nsteps); ///< advance the system for a given number of time steps. Must be called after init()

    /** \brief Send a tagged message to the \c Postprocess rank.
        This is useful to pass special messages, e.g. termination or checkpoint.
     */
    void notifyPostProcess(int tag, int msg) const;

    /// \return a list of all ParticleVector registered objects
    std::vector<ParticleVector*> getParticleVectors() const;

    ParticleVector* getPVbyName     (const std::string& name) const; ///< \return ParticleVector with given name if found, \c nullptr otherwise
    ParticleVector* getPVbyNameOrDie(const std::string& name) const; ///< \return ParticleVector with given name if found, die otherwise
    ObjectVector*   getOVbyName     (const std::string& name) const; ///< \return ObjectVector with given name if found, \c nullptr otherwise
    ObjectVector*   getOVbyNameOrDie(const std::string& name) const; ///< \return ObjectVector with given name if found, die otherwise

    /// \return ParticleVector with the given name if found, \c nullptr otherwise
    std::shared_ptr<ParticleVector> getSharedPVbyName(const std::string& name) const;

    /// \return \c Wall with the given name if found, die otherwise
    Wall* getWallByNameOrDie(const std::string& name) const;

    /** \return the CellList associated to the given ParticleVector, nullptr if there is none
        \param pv The registered ParticleVector

        This method will die if \p pv was not registered
     */
    CellList* gelCellList(ParticleVector* pv) const;

    void startProfiler() const; ///< start the cuda profiler; used for nvprof
    void stopProfiler() const;  ///< end the cuda profiler; used for nvprof

    MPI_Comm getCartComm() const; ///< \return the cartesian communicator of the \c Simulation
    int3 getRank3D() const;       ///< \return the coordinates in the cartesian communicator of the current rank
    int3 getNRanks3D() const;     ///< \return the dimensions of the cartesian communicator

    real getCurrentDt() const;   ///< \return The current time step
    real getCurrentTime() const; ///< \return The current simulation time

    /** \return The largest cut-off radius of all "full" force computation.

        This takes into account the intermediate interactions, e.g. in SDPD
        this will corrspond to the cutoff used for the density + the one from
        the SDPD kernel.
        Useful e.g. to decide the widh of frozen particles in walls.
     */
    real getMaxEffectiveCutoff() const;

    /** \brief dump the task dependency of the simulation in graphML format.
        \param fname The file name to dump the graph to (without extension).
        \param current if \c true, will only dump the current tasks; otherwise, will dump all possible ones.
     */
    void dumpDependencyGraphToGraphML(const std::string& fname, bool current) const;

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    std::vector<std::string> _getExtraDataToExchange(ObjectVector *ov) const;
    std::vector<std::string> _getDataToSendBack(const std::vector<std::string>& extraOut, ObjectVector *ov) const;

    void _prepareCellLists();
    void _prepareInteractions();
    void _prepareBouncers();
    void _prepareWalls();
    void _preparePlugins();
    void _prepareEngines();

    void _execSplitters();

    void _createTasks();
    void _cleanup(); ///< Detach run data from all objects and deallocate run_.

    using MirObject::restart;
    using MirObject::checkpoint;

    void _restartState(const std::string& folder);
    void _checkpointState();

private:
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

    /// Data constructed in init() and used in run().
    std::unique_ptr<RunData> run_;

    const bool gpuAwareMPI_;


    std::map<std::string, int> pvIdMap_;
    std::vector< std::shared_ptr<ParticleVector> > particleVectors_;
    std::vector< ObjectVector* >   objectVectors_;

    MapShared <Bouncer>                bouncerMap_;
    MapShared <Integrator>             integratorMap_;
    MapShared <Interaction>            interactionMap_;
    MapShared <Wall>                   wallMap_;
    MapShared <ObjectBelongingChecker> belongingCheckerMap_;

    std::vector< std::shared_ptr<SimulationPlugin> > plugins;

    std::vector<IntegratorPrototype>          integratorPrototypes_;
    std::vector<InteractionPrototype>         interactionPrototypes_;
    std::vector<WallPrototype>                wallPrototypes_;
    std::vector<CheckWallPrototype>           checkWallPrototypes_;
    std::vector<BouncerPrototype>             bouncerPrototypes_;
    std::vector<BelongingCorrectionPrototype> belongingCorrectionPrototypes_;
    std::vector<SplitterPrototype>            splitterPrototypes_;

    std::map<std::string, std::string> pvsIntegratorMap_;
};

} // namespace mirheo
