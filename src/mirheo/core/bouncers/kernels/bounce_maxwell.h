// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/macros.h>

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

    /** Compute the velocity after bouncing the particle.
        The velocity is chosen such that it is sampled by a Maxwelian
        distribution and has a positive dot product with the wall surface normal.

        \param [in] uOld The velocity of the particle at the previous time step.
        \param [in] uWall The velocity of the wall surface at the collision point.
        \param [in] n The wall surface normal at the collision point.
        \param [in] mass The particle mass.
     */
    __HD__ real3 newVelocity(real3 uOld, real3 uWall, real3 n, real mass) const
    {
        constexpr int maxTries = 50;
        real2 rand1 = Saru::normal2(seed1_, uOld.x, uOld.y);
        real2 rand2 = Saru::normal2(seed2_, uOld.z, uWall.x);

        real3 v = make_real3(rand1.x, rand1.y, rand2.x);

        for (int i = 0; i < maxTries; ++i)
        {
            if (dot(v, n) > 0) break;

            rand1 = Saru::normal2(rand2.x, rand2.y, uOld.x);
            rand2 = Saru::normal2(rand1.x, rand1.x, uOld.y);
            v = make_real3(rand1.x, rand1.y, rand2.x);
        }
        v = normalize(v) * math::sqrt(kBT_ / mass);

        return uWall + v;
    }

private:

    real seed1_{0._r};
    real seed2_{0._r};
    real kBT_;
};

} // namespace mirheo
