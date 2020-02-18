#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/macros.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

class ParticleVector;

/** \brief Apply a constant force independently of the position.
 */
class ForcingTermConstDP
{
public:
    /**\brief Construct a \c ForcingTermConstDP object
       \param [in] extraForce The force to add to every particle
    */
    ForcingTermConstDP(real3 extraForce) :
        extraForce_(extraForce)
    {}

    /**\brief Initialize internal state
       \param [in] pv the \c ParticleVector that will be updated
       \param [in] t Current simulation time

       This method must be called at every time step.
    */
    void setup(__UNUSED ParticleVector* pv, __UNUSED real t)
    {}

    /**\brief Add the additional force to the current one on a particle
       \param [in] original Original force acting on the particle
       \param [in] p Particle on which to apply the additional force
       \return The total force that must be applied to the particle
    */
    __D__ inline real3 operator()(real3 original, __UNUSED Particle p) const
    {
        return extraForce_ + original;
    }

private:
    real3 extraForce_;

    friend MemberVars<ForcingTermConstDP>;
};

MIRHEO_MEMBER_VARS(ForcingTermConstDP, extraForce_);

} // namespace mirheo
