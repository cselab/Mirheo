#include "time_stamp.h"

namespace mirheo
{

bool isTimeEvery(const MirState *state, int dumpEvery)
{
    return state->currentStep % dumpEvery == 0;
}

MirState::StepType getTimeStamp(const MirState *state, int dumpEvery)
{
    return state->currentStep / dumpEvery;
}

} // namespace mirheo
