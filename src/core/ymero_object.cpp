#include "ymero_object.h"

#include <core/logger.h>
#include <core/utils/folders.h>

YmrObject::YmrObject(std::string name) :
    name(name)
{}

YmrObject::~YmrObject() = default;

void YmrObject::checkpoint(MPI_Comm comm, std::string path, int checkpointId) {}
void YmrObject::restart   (MPI_Comm comm, std::string path) {}


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


std::string YmrObject::createCheckpointName(std::string path, std::string identifier, std::string extension) const
{
    auto base = createBaseName(path, name, identifier);
    appendIfNonEmpty(base, extension);
    return base;
}

std::string YmrObject::createCheckpointNameWithId(std::string path, std::string identifier, std::string extension, int checkpointId) const
{
    auto base = createBaseName(path, name, identifier);
    base += "-" + getStrZeroPadded(checkpointId);
    appendIfNonEmpty(base, extension);
    return base;
}

void YmrObject::createCheckpointSymlink(MPI_Comm comm, std::string path, std::string identifier, std::string extension, int checkpointId) const
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


// void YmrObject::advanceCheckpointId(CheckpointIdAdvanceMode mode)
// {
//     if (mode == CheckpointIdAdvanceMode::PingPong)
//         checkpointId = checkpointId xor 1;
//     else
//         ++checkpointId;
// }

YmrSimulationObject::YmrSimulationObject(const YmrState *state, std::string name) :
    YmrObject(name),
    state(state)
{}

YmrSimulationObject::~YmrSimulationObject() = default;
