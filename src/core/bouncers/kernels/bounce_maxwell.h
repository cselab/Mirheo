#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/macros.h>
#include <core/utils/helper_math.h>

// reflection with random scattering
// according to Maxwell distr
class BounceMaxwell
{
public:
    BounceMaxwell(float kBT, float mass, float seed1, float seed2) :
        seed1(seed1),
        seed2(seed2),
        kBT_mass(kBT / mass)
    {}
    
    __D__ float3 newVelocity(__UNUSED float3 uOld, float3 uWall, float3 n) const
    {
        constexpr int maxTries = 50;
        const float2 rand1 = Saru::normal2(seed1, threadIdx.x, blockIdx.x);
        const float2 rand2 = Saru::normal2(seed2, threadIdx.x, blockIdx.x);

        float3 v = make_float3(rand1.x, rand1.y, rand2.x);

        for (int i = 0; i < maxTries; ++i)
        {
            if (dot(v, n) > 0) break;

            const float2 rand3 = Saru::normal2(rand2.y, threadIdx.x, blockIdx.x);
            const float2 rand4 = Saru::normal2(rand3.y, threadIdx.x, blockIdx.x);
            v = make_float3(rand3.x, rand3.y, rand4.x);
        }
        v = normalize(v) * sqrtf(kBT_mass);

        return uWall + v;
    }

private:

    const float seed1, seed2;
    const float kBT_mass;
};
