#pragma once

#include "mirheo_state.h"

#include <mirheo/core/utils/common.h>

#include <mpi.h>
#include <string>

namespace mirheo
{

/**
 * Base class for all the objects of Mirheo
 * Only stores name and provides interface for
 * checkpoint / restart mechanism
 */
class MirObject : public AutoObjectSnapshotTag
{
public:
    MirObject(const std::string& name);
    virtual ~MirObject();

    const std::string& getName() const noexcept {return name_;}
    const char* getCName() const {return name_.c_str();}
    
    virtual void checkpoint(MPI_Comm comm, const std::string& path, int checkPointId);  /// Save handler state
    virtual void restart   (MPI_Comm comm, const std::string& path);  /// Restore handler state

    /** \brief Dump object data, create config and register the object.
        \param [in,out] save The \c Saver object. Provides save context and serialization functions.
      */
    virtual void saveSnapshotAndRegister(Saver& saver);

    std::string createCheckpointName      (const std::string& path, const std::string& identifier, const std::string& extension) const;
    std::string createCheckpointNameWithId(const std::string& path, const std::string& identifier, const std::string& extension, int checkpointId) const;
    void createCheckpointSymlink(MPI_Comm comm, const std::string& path, const std::string& identifier, const std::string& extension, int checkpointId) const;

protected:
    /** Base snapshot function. Sets the `__category` and `__type` special fields.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] category The type category (e.g. "Integrator", "Plugin"...).
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& category, const std::string& typeName);

private:
    const std::string name_;
};

/**
 * Base class for the objects of Mirheo simulation task
 * may additionally store global quantities in the future
 */
class MirSimulationObject : public MirObject
{
public:
    MirSimulationObject(const MirState *state, const std::string& name);

    /** \brief Base constructor. Read the name object from the config.
        \param [in] state The global state of the system.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The object parameters.
     */
    MirSimulationObject(const MirState *state, Loader&, const ConfigObject&);

    ~MirSimulationObject();

    const MirState* getState() const {return state_;}
    virtual void setState(const MirState *state);
    
private:
    const MirState *state_;
};

} // namespace mirheo
