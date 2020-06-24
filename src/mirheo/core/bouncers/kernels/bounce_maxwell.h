// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/macros.h>
#include <mirheo/core/utils/helper_math.h>

#include <random>

namespace mirheo
{

/** \brief Implements reflection with Maxwell scattering.

    This bounce kernel sets the particle velocity to the surface one with an
    additional random term drawed from Maxwell distribution.
    The kernel tries to make the random term have a positive dot product with
    the surface normal.
 */
class BounceMaxwell
{
public:
    /** \brief Construct a BounceMaxwell object
        \param [in] kBT The temperature used to sample the velocity
     */
    BounceMaxwell(real kBT) :
        kBT_(kBT)
    {}

    /** \brief Update internal state, must be called before use.
        \param rng A random number generator.
     */
    void update(std::mt19937& rng)
    {
        std::uniform_real_distribution<real> dis(0._r, 1._r);
        seed1_ = dis(rng);
        seed2_ = dis(rng);
    }

#ifdef __NVCC__
    __device__ real3 newVelocity(__UNUSED real3 uOld, real3 uWall, real3 n, real mass) const
    {
        constexpr int maxTries = 50;
        const real2 rand1 = Saru::normal2(seed1_, threadIdx.x, blockIdx.x);
        const real2 rand2 = Saru::normal2(seed2_, threadIdx.x, blockIdx.x);

        real3 v = make_real3(rand1.x, rand1.y, rand2.x);

        for (int i = 0; i < maxTries; ++i)
        {
            if (dot(v, n) > 0) break;

            const real2 rand3 = Saru::normal2(rand2.y, threadIdx.x, blockIdx.x);
            const real2 rand4 = Saru::normal2(rand3.y, threadIdx.x, blockIdx.x);
            v = make_real3(rand3.x, rand3.y, rand4.x);
        }
        v = normalize(v) * math::sqrt(kBT_ / mass);

        return uWall + v;
    }
#endif

private:

    real seed1_{0._r};
    real seed2_{0._r};
    real kBT_;
};

} // namespace mirheo
