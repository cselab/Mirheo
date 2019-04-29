#pragma once

#include <core/ymero_state.h>

YmrState::StepType getTimeStamp(const YmrState *state, int dumpEvery);
YmrState::StepType getTimeStamp(const YmrState*, float) = delete;

