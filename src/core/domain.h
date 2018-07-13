#pragma once

#ifdef __NVCC__
#define __HD__ __host__ __device__
#else
#define __HD__ 
#endif

#include <cuda_runtime.h>
#include <vector_types.h>
#include <core/utils/helper_math.h>

struct DomainInfo
{
    float3 globalSize, globalStart, localSize;

    inline __HD__ float3 local2global(float3 x) const
    {
        return x + globalStart + 0.5f * localSize;
    }
    inline __HD__ float3 global2local(float3 x) const
    {
        return x - globalStart - 0.5f * localSize;
    }
};
