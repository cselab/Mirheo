#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

class ParticleVector;

/** \brief Interface for assitional force functors.
 */
class ForcingTerm
{
public:
    /**\brief Initialize internal state
       \param [in] pv the \c ParticleVector that will be updated
       \param [in] t Current simulation time

       This method must be called at every time step
    */
    virtual void setup(ParticleVector *pv, real t) = 0;

    /**\brief Add the additional force to the current one on a particle
       \param [in] original Original force acting on the particle
       \param [in] p Particle on which to apply the additional force
       \return The total force that must be applied to the particle

       The return might be the additional term only if that is inended.
       Otherwise, this must be the sum or original and additional term.
    */
    virtual __D__ real3 operator()(real3 original, Particle p) const = 0;
};

} // namespace mirheo
