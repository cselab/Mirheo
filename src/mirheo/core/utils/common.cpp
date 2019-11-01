#include "common.h"

namespace mirheo
{

CheckpointInfo::CheckpointInfo(int every, const std::string& folder,
                               CheckpointIdAdvanceMode mode) :
    every(every),
    folder(folder),
    mode(mode)
{}

} // namespace mirheo
