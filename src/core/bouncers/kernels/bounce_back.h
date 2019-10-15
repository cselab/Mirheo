#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/macros.h>

#include <random>

class BounceBack
{
public:
    BounceBack() = default;

    void update(__UNUSED std::mt19937& rng) {}

#ifdef __NVCC__
    __device__  float3 newVelocity(float3 uOld, float3 uWall, __UNUSED float3 n, __UNUSED float mass) const
    {
        return uWall - (uOld - uWall);
    }
#endif
};
