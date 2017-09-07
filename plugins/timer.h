#pragma once

#include <chrono>

template<typename Ratio = std::milli>
class Timer
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start, _end;

	std::chrono::time_point<std::chrono::high_resolution_clock> none =
	std::chrono::time_point<std::chrono::high_resolution_clock>::min();


public:

    inline Timer()
    {
    	static_assert(std::chrono::__is_ratio<Ratio>::value, "timer must be a specialized with ratio");

        _start = none;
        _end   = none;
    }

    inline void start()
    {
        _start = std::chrono::high_resolution_clock::now();
        _end   = none;
    }

    inline void stop()
    {
        _end = std::chrono::high_resolution_clock::now();
    }

    inline double elapsed()
    {
        if (_end == none) _end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration <double, Ratio>(_end - _start).count();
    }

    inline double elapsedAndReset()
    {
        if (_end == none) _end = std::chrono::high_resolution_clock::now();

        double t = std::chrono::duration <double, Ratio>(_end - _start).count();

        _start = _end;
        _end = none;
        return t;
    }
};
