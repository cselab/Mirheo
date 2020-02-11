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
class MirObject
{
public:
    MirObject(const std::string& name);
    virtual ~MirObject();

    const std::string& getName() const noexcept {return name_;}
    const char* getCName() const {return name_.c_str();}
    
    virtual void checkpoint(MPI_Comm comm, const std::string& path, int checkPointId);  /// Save handler state
    virtual void restart   (MPI_Comm comm, const std::string& path);  /// Restore handler state

    /// Dump object data, create config, register the object and returns its refstring.
    virtual void saveSnapshotAndRegister(Saver&);

    std::string createCheckpointName      (const std::string& path, const std::string& identifier, const std::string& extension) const;
    std::string createCheckpointNameWithId(const std::string& path, const std::string& identifier, const std::string& extension, int checkpointId) const;
    void createCheckpointSymlink(MPI_Comm comm, const std::string& path, const std::string& identifier, const std::string& extension, int checkpointId) const;

protected:
    ConfigObject _saveSnapshot(Saver&, const std::string& category, const std::string& typeName);

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
    MirSimulationObject(const MirState *state, Loader&, const ConfigObject&);
    ~MirSimulationObject();

    const MirState* getState() const {return state_;}
    virtual void setState(const MirState *state);
    
private:
    const MirState *state_;
};

// Common saver to the template saver below. This way we don't need the
// definition of ConfigValue here.
struct MirObjectLoadSave
{
    // Automatically adds `name` key to the returned dictionary.
    static ConfigValue save(Saver&, MirObject& obj);
};

/// TypeLoadSave specialization for MirObject and derived classes.
/// Note: this will also match derived class where MirObject is not the first
/// base class. The registration of these objects will fail.
template <typename T>
struct TypeLoadSave<T, std::enable_if_t<std::is_base_of<MirObject, T>::value>>
    : MirObjectLoadSave
{ };

} // namespace mirheo
