#include "time_stamp.h"

bool isTimeEvery(const YmrState *state, int dumpEvery)
{
    return state->currentStep % dumpEvery == 0;
}

YmrState::StepType getTimeStamp(const YmrState *state, int dumpEvery)
{
    return state->currentStep / dumpEvery - 1;
}
