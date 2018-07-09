#pragma once

#ifdef __NVCC__

#include <core/utils/cuda_common.h>

struct DomainInfo
{
    float3 globalSize, globalStart, localSize;

    inline __host__ __device__ float3 local2global(float3 x) const
    {
        return x + globalStart + 0.5f * localSize;
    }
    inline __host__ __device__ float3 global2local(float3 x) const
    {
        return x - globalStart - 0.5f * localSize;
    }
};

#else

struct DomainInfo
{
    float3 globalSize, globalStart, localSize;

    inline float3 local2global(float3 x) const
    {
        return { x.x + globalStart.x + 0.5f * localSize.x,
                 x.y + globalStart.y + 0.5f * localSize.y,
                 x.z + globalStart.z + 0.5f * localSize.z  };
    }
    inline float3 global2local(float3 x) const
    {
        return { x.x - globalStart.x - 0.5f * localSize.x,
                 x.y - globalStart.y - 0.5f * localSize.y,
                 x.z - globalStart.z - 0.5f * localSize.z  };
    }
};

#endif
