#pragma once

#include "interface.h"

#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/macros.h>

namespace mirheo
{

class ParticleVector;

/** \brief Applies constant force #extraForce to every particle
 */
class Forcing_ConstDP: public ForcingTerm
{
public:
    Forcing_ConstDP(real3 extraForce) :
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
