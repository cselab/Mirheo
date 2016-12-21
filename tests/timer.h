#pragma once

#include <chrono>

class Timer
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start, _end;

	std::chrono::time_point<std::chrono::high_resolution_clock> none =
	std::chrono::time_point<std::chrono::high_resolution_clock>::min();

public:

    inline Timer()
    {
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

    inline int64_t elapsed()
    {
        if (_end == none) _end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration <long int, std::nano>(_end - _start).count();
    }

    inline int64_t elapsedAndReset()
    {
        if (_end == none) _end = std::chrono::high_resolution_clock::now();

        int64_t t = std::chrono::duration <int64_t, std::nano>(_end - _start).count();

        _start = _end;
        _end = none;
        return t;
    }
};
