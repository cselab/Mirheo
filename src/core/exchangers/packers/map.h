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
    int low = 0, hig = nBuffers;
    while (hig > low+1)
    {
        int mid = (low + hig) / 2;
        bool moveUp = tid >= offsets[mid];
        low = moveUp ? mid : low;
        hig = moveUp ? hig : mid;
    }
    return low;
}
