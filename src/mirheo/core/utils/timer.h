// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <chrono>

namespace mirheo
{

/** \brief Measure wall time: profiling tool.
    \tparam Ratio a type of std::ratio, used to set the units of time durations (e.g. s, ms, us)
 */
template<typename Ratio>
class Timer
{
public:
    /// Construct a timer object
    Timer() :
        start_ {_none()},
        end_   {_none()}
    {
        static_assert(std::chrono::__is_ratio<Ratio>::value, "timer must be specialized with ratio");
    }

    /// Start the wall clock
    void start()
    {
        start_ = Clock::now();
        end_   = _none();
    }

    /// Stop the wall clock and record the wall time
    void stop()
    {
        end_ = Clock::now();
    }

    /** Compute the elapsed time between the previous start() and stop() calls.
        If no stop() was called, computes the duration between start() and the current call.
        This call does not reset the start time stamp, so it can be called later again.
        \note start() must have been called at least once before this called.

        \return The ellapsed time
    */
    double elapsed()
    {
        if (end_ == _none()) end_ = Clock::now();
        return std::chrono::duration <double, Ratio>(end_ - start_).count();
    }

    /** \return see ellapsed()

        This will also reset the start timestamp to the current time.
     */
    double elapsedAndReset()
    {
        const double t = elapsed();
        start_ = end_;
        end_ = _none();
        return t;
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    using Time = std::chrono::time_point<Clock>;

    static constexpr Time _none() {
        return Time::min();
    }

    Time start_;
    Time end_;
};

using uTimer = Timer<std::micro>; ///< timer which measures time in us
using mTimer = Timer<std::milli>; ///< timer which measures time in ms
using sTimer = Timer<std::ratio<1, 1>>; ///< timer which measures time in s

} // namespace mirheo
