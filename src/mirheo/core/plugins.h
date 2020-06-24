// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/mirheo_object.h>

#include <mpi.h>
#include <vector>

namespace mirheo
{

class Simulation;

/** \brief Base class to represent a Plugin.

    Plugins are functionalities that are not required to run a simulation.
    Each plugin must have a SimulationPlugin derived class, and, optionally,
    a compatible PostprocessPlugin derived class.
    The latter is used to perform potentially expensive work asynchronously
    while the simulation is running (e.g. I/O).
 */
class Plugin
{
public:
    /// default constructor
    Plugin();
    virtual ~Plugin();

    /// Used to communicate initial information between a SimulationPlugin and a PostprocessPlugin.
    /// Does not do anything by default.
    virtual void handshake();

    /** \brief Set the tag that will be used internally to communicate between
        SimulationPlugin and a PostprocessPlugin.
        \param tag The tag, must be unique (all plugins using the same intercommunicator must have a different tag, see _setup())

        Must be called before any other methods.
    */
    void setTag(int tag);

protected:
    /** Setup the internal state from the given MPI communicators.
        Must be called before any other method of the class.
        \param comm The communicator that holds all simulation or postprocess ranks. Will be duplicated.
        \param interComm The communicator to communicate between simulation and postprocess ranks.
                Will not be duplicated, so the user must ensure that it stays allocated while the Plugin is alive.
     */
    void _setup(const MPI_Comm& comm, const MPI_Comm& interComm);
    int _sizeTag() const; ///< generate a tag to communicate the size of a message
    int _dataTag() const; ///< generate a tag to communicate the content of a message

private:
    void _checkTag() const; ///< die if the tag has not been set.

protected:
    MPI_Comm comm_;      ///< The communicator shared by all simulation or postprocess ranks.
    MPI_Comm interComm_; ///< The communicator used to communicate between simulation and postprocess ranks.
    int rank_;   ///< rank id within comm_
    int nranks_; ///< number of ranks in comm_

private:
    static constexpr int invalidTag = -1;
    int tag_ {invalidTag};
};

/** \brief Base class for the simulation side of a \c Plugin.

    A simulation plugin is able to modify the state of the simulation.
    Depending on its action, one of the "hooks" (e.g. beforeCellLists())
    must be overriden (by default they do not do anything).

    If a plugin needs reference to objects held by the simulation, it must
    be saved in its internal structure at setup() time.
 */
class SimulationPlugin : public Plugin, public MirSimulationObject
{
public:
    /** \brief Construct a SimulationPlugin
        \param state The global simulation state
        \param name the name of the plugin (must be the same as that of the postprocess plugin)
     */
    SimulationPlugin(const MirState *state, const std::string& name);
    virtual ~SimulationPlugin();

    /** \return \c true if this plugin needs a postprocess side; \c false otherwise.
        \note The plugin can have a postprocess side but not need it.
    */
    virtual bool needPostproc() = 0;

    /** \brief setup the internal state of the SimulationPlugin.
        \param simulation The simulation to which the plugin is registered.
        \param comm Contains all simulation ranks
        \param interComm used to communicate with the postprocess ranks

        This method must be called before any of the hooks of the plugin.
        This is the place to fetch reference to objects held by the simulation.
     */
    virtual void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm);

    virtual void beforeCellLists            (cudaStream_t stream); ///< hook before building the cell lists
    virtual void beforeForces               (cudaStream_t stream); ///< hook before computing the forces and after the cell lists are created
    virtual void beforeIntegration          (cudaStream_t stream); ///< hook before integrating the particle vectors but after the forces are computed
    virtual void afterIntegration           (cudaStream_t stream); ///< hook after the ObjectVector objects are integrated but before redistribution and bounce back
    virtual void beforeParticleDistribution (cudaStream_t stream); ///< hook before redistributing ParticleVector objects and after bounce

    /** \brief Pack and send data to the postprocess rank.
        Happens between beforeForces() and beforeIntegration().
        \note This may happens while computing the forces.
     */
    virtual void serializeAndSend (cudaStream_t stream);

    virtual void finalize(); ///< hook that happens once at the end of the simulation loop

protected:
    /// wait for the previous send request to complete
    void _waitPrevSend();

    /// post an asynchronous send for the given data to the postprocess rank
    void _send(const std::vector<char>& data);
    /// see send()
    void _send(const void *data, size_t sizeInBytes);

    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    int localSendSize_;
    MPI_Request sizeReq_;
    MPI_Request dataReq_;
};

/** \brief Base class for the postprocess side of a \c Plugin.

    A postprocess plugin can only receive information from its associated SimulationPlugin.
    The use of such class is to wait for a message and then deserialize it (where additional
    actions may be performed, such as I/O).
 */
class PostprocessPlugin : public Plugin, public MirObject
{
public:
    /** \brief Construct a PostprocessPlugin
        \param name the name of the plugin (must be the same as that of the associated simulation plugin)
     */
    PostprocessPlugin(const std::string& name);
    virtual ~PostprocessPlugin();

    /** \brief setup the internal state of the PostprocessPlugin.
        \param comm Contains all postprocess ranks
        \param interComm used to communicate with the simulation ranks

        This method must be called before any other function call.
     */
    virtual void setup(const MPI_Comm& comm, const MPI_Comm& interComm);

    /// Post an asynchronous receive request to get a message from the associated SimulationPlugin
    void recv();

    /// wait for the completion of the asynchronous receive request. Must be called after recv() and before deserialize().
    MPI_Request waitData();

    /// Perform the action implemented by the plugin using the data received from the SimulationPlugin.
    virtual void deserialize() = 0;

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

    std::vector<char> data_; ///< will hold the data sent by the associated SimulationPlugin
private:
    int size_; ///< size of the recv data
};

} // namespace mirheo
