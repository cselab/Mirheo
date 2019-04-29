#include "time_stamp.h"

YmrState::StepType getTimeStamp(const YmrState *state, int dumpEvery)
{
    return state->currentStep / dumpEvery;
}
