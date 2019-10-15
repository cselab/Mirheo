#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/macros.h>
#include <core/utils/helper_math.h>

#include <random>

// reflection with random scattering
// according to Maxwell distr
class BounceMaxwell
{
public:
    BounceMaxwell(float kBT) :
        kBT(kBT)
    {}

    void update(std::mt19937& rng)
    {
        std::uniform_real_distribution<float> dis(0.f, 1.f);
        seed1 = dis(rng);
        seed2 = dis(rng);
    }

#ifdef __NVCC__
    __device__ float3 newVelocity(__UNUSED float3 uOld, float3 uWall, float3 n, float mass) const
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
        v = normalize(v) * sqrtf(kBT / mass);

        return uWall + v;
    }
#endif

private:

    float seed1{0.f};
    float seed2{0.f};
    const float kBT;
};
