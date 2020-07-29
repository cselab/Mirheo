// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/mirheo_state.h>

namespace mirheo
{
/** Check if a dump should occur at the current time step.
    \param [in] state The current state of the simulation.
    \param [in] dumpEvery  The number of steps between two dumps.
    \return \c true if the current step is a dump time; \c false otherwise.
 */
bool isTimeEvery(const MirState *state, int dumpEvery);
#ifndef DOXYGEN_SHOULD_SKIP_THIS
bool isTimeEvery(const MirState *state, real dumpEvery) = delete;
#endif

/** Get the dump stamp from current time and dump frequency.
    \param [in] state The current state of the simulation.
    \param [in] dumpEvery The number of steps between two dumps.
    \return The dump stamp.
 */
MirState::StepType getTimeStamp(const MirState *state, int dumpEvery);
#ifndef DOXYGEN_SHOULD_SKIP_THIS
MirState::StepType getTimeStamp(const MirState*, real) = delete;
#endif

} // namespace mirheo
