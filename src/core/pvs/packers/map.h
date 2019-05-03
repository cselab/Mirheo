#pragma once

#include <core/utils/cpu_gpu_defines.h>

#include <cstdint>

struct __align__(4) MapEntry
{
    // 27 < 2^5 buffers max
    static constexpr int bufWidth = 5;
    static constexpr int bufShift = 32 - bufWidth;
    static constexpr uint32_t maskAll = 0xffffffff; 
    static constexpr uint32_t maskId  = (maskAll << bufWidth) >> bufWidth;
    static constexpr uint32_t maskBuf = ~maskId;
    
    uint32_t i;

    __HD__ inline uint32_t getId()    const {return i & maskId;}
    __HD__ inline uint32_t getBufId() const {return i >> bufShift;}
    
    __HD__ inline void setId(uint32_t id)
    {
        uint32_t bufInfo = maskBuf & i;
        i = bufInfo | id;
    }

    __HD__ inline void setBufId(uint32_t bufId)
    {
        uint32_t  idInfo = maskId & i;
        uint32_t bufInfo = bufId << bufShift;
        i = bufInfo | idInfo;
    }
};

inline __HD__ int dispatchThreadsPerBuffer(int nBuffers, const int *offsets, int tid)
{
    int lo = 0, hi = nBuffers;
    while (hi > lo+1)
    {
        int m = (lo + hi) / 2;
        if (tid >= offsets[m]) lo = m;
        else hi = m;
    }
    return lo;
}
