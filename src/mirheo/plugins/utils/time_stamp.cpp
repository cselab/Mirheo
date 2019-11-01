#include "time_stamp.h"

bool isTimeEvery(const MirState *state, int dumpEvery)
{
    return state->currentStep % dumpEvery == 0;
}

MirState::StepType getTimeStamp(const MirState *state, int dumpEvery)
{
    return state->currentStep / dumpEvery;
}
