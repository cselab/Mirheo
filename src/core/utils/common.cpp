#include "common.h"

CheckpointInfo::CheckpointInfo(int every, const std::string& folder,
                               CheckpointIdAdvanceMode mode) :
    every(every),
    folder(folder),
    mode(mode)
{}
