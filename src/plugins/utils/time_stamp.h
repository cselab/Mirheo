#pragma once

#include <core/mirheo_state.h>

bool isTimeEvery(const MirState *state, int dumpEvery);
bool isTimeEvery(const MirState *state, float dumpEvery) = delete;

MirState::StepType getTimeStamp(const MirState *state, int dumpEvery);
MirState::StepType getTimeStamp(const MirState*, float) = delete;

