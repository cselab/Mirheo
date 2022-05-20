// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

class ParticleVector;

/** \brief No forcing term.
 */
class ForcingTermNone
{
public:
    /**\brief Initialize internal state
       \param [in] pv the ParticleVector that will be updated
       \param [in] t Current simulation time

       This method must be called at every time step
    */
    void setup(__UNUSED ParticleVector *pv, __UNUSED real t)
    {}

    /**\brief Add the additional force to the current one on a particle
       \param [in] original Original force acting on the particle
       \param [in] p Particle on which to apply the additional force
       \return The total force that must be applied to the particle
    */
    __D__ inline real3 operator()(real3 original, __UNUSED Particle p) const
    {
        return original;
    }
};

} // namespace mirheo
