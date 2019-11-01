#pragma once

#include <chrono>

namespace mirheo
{

template<typename Ratio>
class Timer
{
    using Clock = std::chrono::high_resolution_clock;
    
private:
    std::chrono::time_point<Clock> _start, _end;

    std::chrono::time_point<Clock> none =
    std::chrono::time_point<Clock>::min();


public:

    inline Timer()
    {
        static_assert(std::chrono::__is_ratio<Ratio>::value, "timer must be specialized with ratio");

        _start = none;
        _end   = none;
    }

    inline void start()
    {
        _start = Clock::now();
        _end   = none;
    }

    inline void stop()
    {
        _end = Clock::now();
    }

    inline double elapsed()
    {
        if (_end == none) _end = Clock::now();

        return std::chrono::duration <double, Ratio>(_end - _start).count();
    }

    inline double elapsedAndReset()
    {
        if (_end == none) _end = Clock::now();

        double t = std::chrono::duration <double, Ratio>(_end - _start).count();

        _start = _end;
        _end = none;
        return t;
    }
};

using uTimer = Timer<std::micro>;
using mTimer = Timer<std::milli>;
using sTimer = Timer<std::ratio<1, 1>>;

} // namespace mirheo
