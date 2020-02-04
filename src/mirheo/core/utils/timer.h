#pragma once

#include <chrono>

namespace mirheo
{

template<typename Ratio>
class Timer
{
public:

    Timer() :
        start_ {none_},
        end_   {none_}
    {
        static_assert(std::chrono::__is_ratio<Ratio>::value, "timer must be specialized with ratio");
    }

    void start()
    {
        start_ = Clock::now();
        end_   = none_;
    }

    void stop()
    {
        end_ = Clock::now();
    }

    double elapsed()
    {
        if (end_ == none_) end_ = Clock::now();
        return std::chrono::duration <double, Ratio>(end_ - start_).count();
    }

    double elapsedAndReset()
    {
        const double t = elapsed();
        start_ = end_;
        end_ = none_;
        return t;
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    using Time = std::chrono::time_point<Clock>;
    
    Time start_;
    Time end_;

    static constexpr Time none_ {Time::min()};
};

using uTimer = Timer<std::micro>;
using mTimer = Timer<std::milli>;
using sTimer = Timer<std::ratio<1, 1>>;

} // namespace mirheo
