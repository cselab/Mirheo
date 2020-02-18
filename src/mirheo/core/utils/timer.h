#pragma once

#include <chrono>

namespace mirheo
{

template<typename Ratio>
class Timer
{
public:

    Timer() :
        start_ {_none()},
        end_   {_none()}
    {
        static_assert(std::chrono::__is_ratio<Ratio>::value, "timer must be specialized with ratio");
    }

    void start()
    {
        start_ = Clock::now();
        end_   = _none();
    }

    void stop()
    {
        end_ = Clock::now();
    }

    double elapsed()
    {
        if (end_ == _none()) end_ = Clock::now();
        return std::chrono::duration <double, Ratio>(end_ - start_).count();
    }

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

using uTimer = Timer<std::micro>;
using mTimer = Timer<std::milli>;
using sTimer = Timer<std::ratio<1, 1>>;

} // namespace mirheo
