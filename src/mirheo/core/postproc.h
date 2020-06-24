// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/mirheo_object.h>
#include <mirheo/core/plugins.h>

#include <memory>
#include <mpi.h>

namespace mirheo
{

/** \brief Manage post processing tasks (see \c Plugin) related to a \c Simulation.

    There must be exactly one \c Postprocess rank per \c Simulation rank or no \c Postprocess rank at all.
    All \c Plugin objects must be registered and set before calling init() and run().
    This can be instantiated on ranks that have no access to GPUs.

    The run() method consists in waiting for messages incoming from the simulation ranks and execute the
    registered plugins functions with that data.
 */
class Postprocess : MirObject
{
public:
    /** \brief Construct a \c Postprocess object
        \param comm a communicator that holds all postprocessing ranks.
        \param interComm An inter communicator to communicate with the \c Simulation ranks.
        \param checkpointInfo Checkpoint configuratoin.
     */
    Postprocess(MPI_Comm& comm, MPI_Comm& interComm, const CheckpointInfo& checkpointInfo);
    ~Postprocess();

    /** \brief Register a plugin to this object.
        \param plugin The plugin to register
        \param tag a tag that is unique for each registered plugin
        \note The SimulationPlugin counterpart of the registered PostprocessPlugin must be registered on the simulation side.
     */
    void registerPlugin(std::shared_ptr<PostprocessPlugin> plugin, int tag);

    /// Setup all registered plugins. Must be called before run()
    void init();
    /// Start the postprocess. Will run until a termination notification is sent by the simulation.
    void run();

    /** \brief Restore the state from checkpoint information.
        \param folder The path containing the checkpoint files
     */
    void restart(const std::string& folder);

    /** \brief Dump the state of all postprocess plugins to the checkpoint folder
        \param checkpointId The index of the dump, used to name the files.
     */
    void checkpoint(int checkpointId);

    /** \brief Save snapshot of the simulation setup and data.
        \param path Target folder.
      */
    void snapshot(const std::string& path);

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    MPI_Request _listenSimulation(int tag, int *msg) const;

    using MirObject::restart;
    using MirObject::checkpoint;

private:
    friend Saver;

    MPI_Comm comm_;
    MPI_Comm interComm_;

    std::vector< std::shared_ptr<PostprocessPlugin> > plugins_;

    std::string checkpointFolder_;
    CheckpointMechanism checkpointMechanism_;
};

} // namespace mirheo
