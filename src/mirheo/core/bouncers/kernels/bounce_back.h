#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
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

#ifdef __NVCC__
    __device__  real3 newVelocity(real3 uOld, real3 uWall, __UNUSED real3 n, __UNUSED real mass) const
    {
        return uWall - (uOld - uWall);
    }
#endif
};

} // namespace mirheo
