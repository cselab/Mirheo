// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/macros.h>

#include <random>

namespace mirheo
{

/** \brief Implements bounce-back reflection.
    This bounce kernel reverses the velocity of the particle in the frame of reference of the surface.
 */
class BounceBack
{
public:
    BounceBack() = default;

    /**
        Does nothing, just to be consistent with the inteface
     */
    void update(__UNUSED std::mt19937& rng) {}

    /** Compute the velocity after bouncing the particle.
        The velocity is chosen such that the average between the new and old
        velocities of the particle is that of the wall surface at the collision point.

        \param [in] uOld The velocity of the particle at the previous time step.
        \param [in] uWall The velocity of the wall surface at the collision point.
        \param [in] n The wall surface normal at the collision point.
        \param [in] mass The particle mass.
     */
    __HD__ real3 newVelocity(real3 uOld, real3 uWall, __UNUSED real3 n, __UNUSED real mass) const
    {
        return uWall - (uOld - uWall);
    }
};

} // namespace mirheo
