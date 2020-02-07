#pragma once

#include "interface.h"

#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

class ParticleVector;

/** \brief Apply a constant force independently of the position.
    \ingroup Integrators
 */
class ForcingTermConstDP: public ForcingTerm
{
public:
    /**\brief Construct a \c ForcingTermConstDP object
       \param [in] extraForce The force to add to every particle
    */
    ForcingTermConstDP(real3 extraForce) :
        extraForce_(extraForce)
    {}

    void setup(__UNUSED ParticleVector* pv, __UNUSED real t) override
    {}

    __D__ inline real3 operator()(real3 original, __UNUSED Particle p) const override
    {
        return extraForce_ + original;
    }

private:
    real3 extraForce_;
};

} // namespace mirheo
