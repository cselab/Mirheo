#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <core/utils/helper_math.h>
#include <core/utils/cpu_gpu_defines.h>

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
