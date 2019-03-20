#include "ymero_object.h"

#include <core/utils/folders.h>

YmrObject::YmrObject(std::string name) :
    name(name)
{}

YmrObject::~YmrObject() = default;

void YmrObject::checkpoint(MPI_Comm comm, std::string path) {}
void YmrObject::restart   (MPI_Comm comm, std::string path) {}

std::string YmrObject::createCheckpointName(std::string path, std::string identifier) const
{
    auto base = path + "/" + name;

    if (identifier != "")
        base += "." + identifier;
    
    return base + "-" + getStrZeroPadded(checkpointId);
}

void YmrObject::advanceCheckpointId(CheckpointIdAdvanceMode mode)
{
    if (mode == CheckpointIdAdvanceMode::PingPong)
        checkpointId = checkpointId xor 1;
    else
        ++checkpointId;
}

YmrSimulationObject::YmrSimulationObject(const YmrState *state, std::string name) :
    YmrObject(name),
    state(state)
{}

YmrSimulationObject::~YmrSimulationObject() = default;
