#include "mirheo_object.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/config.h>

namespace mirheo
{

MirObject::MirObject(const std::string& name) :
    name_(name)
{
    debug4("Creating object '%s'", name_.c_str());
}

MirObject::~MirObject()
{
    debug4("Destroying object '%s'", name_.c_str());
}

void MirObject::checkpoint(__UNUSED MPI_Comm comm, __UNUSED const std::string& path, __UNUSED int checkpointId) {}
void MirObject::restart   (__UNUSED MPI_Comm comm, __UNUSED const std::string& path) {}

void MirObject::saveSnapshotAndRegister(Dumper& dumper)
{
    // This will always trigger a /function not implemented/ runtime error,
    // because MirObject is effectively an abstract class.
    dumper.registerObject<MirObject>(
            this, _saveSnapshot(dumper, "UnknownCategory", "MirObject"));
}

ConfigObject MirObject::_saveSnapshot(Dumper& dumper, const std::string& category, const std::string& typeName)
{
    ConfigObject config;
    // "Unsafe" == skip checking whether the key is already in use.
    config.unsafe_insert("__category", dumper(category));
    config.unsafe_insert("__type", dumper(typeName));
    config.unsafe_insert("name", dumper(name_));
    return config;
}


static void appendIfNonEmpty(std::string& base, const std::string& toAppend)
{
    if (toAppend != "")
        base += "." + toAppend;
}

static std::string createBaseName(const std::string& path,
                                  const std::string& name,
                                  const std::string& identifier)
{
    auto base = path + "/" + name;
    appendIfNonEmpty(base, identifier);
    return base;    
}


std::string MirObject::createCheckpointName(const std::string& path, const std::string& identifier, const std::string& extension) const
{
    std::string base = createBaseName(path, getName(), identifier);
    appendIfNonEmpty(base, extension);
    return base;
}

std::string MirObject::createCheckpointNameWithId(const std::string& path, const std::string& identifier, const std::string& extension, int checkpointId) const
{
    auto base = createBaseName(path, getName(), identifier);
    base += "-" + getStrZeroPadded(checkpointId);
    appendIfNonEmpty(base, extension);
    return base;
}

void MirObject::createCheckpointSymlink(MPI_Comm comm, const std::string& path, const std::string& identifier, const std::string& extension, int checkpointId) const
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    if (rank == 0) {
        const std::string lnname = createCheckpointName      (path, identifier, extension);
        const std::string  fname = createCheckpointNameWithId(path, identifier, extension, checkpointId);
        const std::string command = "ln -f " + fname + " " + lnname;
        
        if ( system(command.c_str()) != 0 )
            error("Could not create symlink '%s' for checkpoint file '%s'",
                  lnname.c_str(), fname.c_str());
    }    
}


MirSimulationObject::MirSimulationObject(const MirState *state, const std::string& name) :
    MirObject(name),
    state_(state)
{}
MirSimulationObject::MirSimulationObject(const MirState *state, Undumper&, const ConfigObject& config) :
    MirSimulationObject(state, config["name"])
{}

MirSimulationObject::~MirSimulationObject() = default;

void MirSimulationObject::setState(const MirState *state)
{
    state_ = state;
}

ConfigValue ConfigMirObjectDumper::dump(Dumper& dumper, MirObject& obj)
{
    if (!dumper.isObjectRegistered(&obj))
        obj.saveSnapshotAndRegister(dumper);
    return dumper.getObjectRefString(&obj);
}

} // namespace mirheo
