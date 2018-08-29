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

    template <typename real3>
    inline __HD__ bool inSubDomain(real3 xg) const
    {
        return (globalStart.x <= xg.x) && (xg.x < (globalStart.x + localSize.x))
            && (globalStart.y <= xg.y) && (xg.y < (globalStart.y + localSize.y))
            && (globalStart.z <= xg.z) && (xg.z < (globalStart.z + localSize.z));
    }    
};
