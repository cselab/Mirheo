#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/macros.h>

class BounceBack
{
public:
    BounceBack() = default;
    
    __D__  float3 newVelocity(float3 uOld, float3 uWall, __UNUSED float3 n) const
    {
        return uWall - (uOld - uWall);
    }
};
