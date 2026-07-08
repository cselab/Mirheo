// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "mirheo_state.h"

#include <mirheo/core/utils/common.h>

#include <mpi.h>
#include <string>

/// \brief Common namespace for all Mirheo code.
namespace mirheo
{

/** \brief Base class for all the objects of Mirheo
 */
class MirObject
{
public:
    /** \brief Construct a MirObject object.
        \param [in] name Name of the object.
     */
    MirObject(const std::string& name);
    virtual ~MirObject();

    /// \brief Return the name of the object.
    const std::string& getName() const noexcept {return name_;}

    /// \brief Return the name of the object in c style. Useful for printf.
    const char* getCName() const {return name_.c_str();}

    /** \brief Save the state of the object on disk.
        \param [in] comm MPI communicator to perform the I/O.
        \param [in] path The directory path to store the object state.
        \param [in] checkPointId The id of the dump.
     */
    virtual void checkpoint(MPI_Comm comm, const std::string& path, int checkPointId);

    /** \brief Load the state of the object from the disk.
        \param [in] comm MPI communicator to perform the I/O.
        \param [in] path The directory path to store the object state.
     */
    virtual void restart(MPI_Comm comm, const std::string& path);

    /** \brief Helper function to create file name for checkpoint/restart.
        \param [in] path The checkpoint/restart directory.
        \param [in] identifier An additional identifier, ignored if empty.
        \param [in] extension File extension.
        \return The file name.
     */
    std::string createCheckpointName(const std::string& path, const std::string& identifier, const std::string& extension) const;

    /** \brief Helper function to create file name for checkpoint/restart with a given Id.
        \param [in] path The checkpoint/restart directory.
        \param [in] identifier An additional identifier, ignored if empty.
        \param [in] extension File extension.
        \param [in] checkpointId Dump Id.
        \return The file name.
    */
    std::string createCheckpointNameWithId(const std::string& path, const std::string& identifier, const std::string& extension, int checkpointId) const;

    /** \brief Create a symlink with a name with no id to the file with a given id.
        \param [in] comm MPI communicator to perform the I/O.
        \param [in] path The checkpoint/restart directory.
        \param [in] identifier An additional identifier, ignored if empty.
        \param [in] extension File extension.
        \param [in] checkpointId Dump Id.
    */
    void createCheckpointSymlink(MPI_Comm comm, const std::string& path, const std::string& identifier, const std::string& extension, int checkpointId) const;

private:
    const std::string name_; ///< Name of the object.
};

/** \brief Base class for the objects of Mirheo simulation task.
    Contains global information common to all objects.
*/
class MirSimulationObject : public MirObject
{
public:
    /** \brief Construct a MirSimulationObject object.
        \param [in] name Name of the object.
        \param [in] state State of the simulation.
     */
    MirSimulationObject(const MirState *state, const std::string& name);

    ~MirSimulationObject();

    /// \brief Return the simulation state.
    const MirState* getState() const {return state_;}
    /// \brief Set the simulation state.
    virtual void setState(const MirState *state);

private:
    const MirState *state_; ///< Global simulation state shared with other Mirheo objects.
};

} // namespace mirheo
