#include "mirheo_object.h"

#include <core/logger.h>
#include <core/utils/folders.h>

MirObject::MirObject(std::string name) :
    name(name)
{}

MirObject::~MirObject() = default;

void MirObject::checkpoint(MPI_Comm comm, std::string path, int checkpointId) {}
void MirObject::restart   (MPI_Comm comm, std::string path) {}


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


std::string MirObject::createCheckpointName(std::string path, std::string identifier, std::string extension) const
{
    auto base = createBaseName(path, name, identifier);
    appendIfNonEmpty(base, extension);
    return base;
}

std::string MirObject::createCheckpointNameWithId(std::string path, std::string identifier, std::string extension, int checkpointId) const
{
    auto base = createBaseName(path, name, identifier);
    base += "-" + getStrZeroPadded(checkpointId);
    appendIfNonEmpty(base, extension);
    return base;
}

void MirObject::createCheckpointSymlink(MPI_Comm comm, std::string path, std::string identifier, std::string extension, int checkpointId) const
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    if (rank == 0) {
        std::string lnname = createCheckpointName      (path, identifier, extension);
        std::string  fname = createCheckpointNameWithId(path, identifier, extension, checkpointId);
        std::string command = "ln -f " + fname + " " + lnname;
        
        if ( system(command.c_str()) != 0 )
            error("Could not create symlink '%s' for checkpoint file '%s'",
                  lnname.c_str(), fname.c_str());
    }    
}


MirSimulationObject::MirSimulationObject(const MirState *state, std::string name) :
    MirObject(name),
    state(state)
{}

MirSimulationObject::~MirSimulationObject() = default;
