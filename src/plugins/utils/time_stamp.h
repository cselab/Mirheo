#pragma once

#include <core/ymero_state.h>

bool isTimeEvery(const YmrState *state, int dumpEvery);
bool isTimeEvery(const YmrState *state, float dumpEvery) = delete;

YmrState::StepType getTimeStamp(const YmrState *state, int dumpEvery);
YmrState::StepType getTimeStamp(const YmrState*, float) = delete;

