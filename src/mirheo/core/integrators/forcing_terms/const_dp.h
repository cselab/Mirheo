#pragma once

#include <mirheo/core/datatypes.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/macros.h>
#include <mirheo/core/utils/reflection.h>

namespace mirheo
{

class ParticleVector;

/**
 * Applies constant force #extraForce to every particle
 */
class Forcing_ConstDP
{
public:
    Forcing_ConstDP(real3 extraForce) : extraForce(extraForce) {}
    void setup(__UNUSED ParticleVector* pv, __UNUSED real t) {}

    __D__ inline real3 operator()(real3 original, __UNUSED Particle p) const
    {
        return extraForce + original;
    }

private:
    real3 extraForce;

    friend MemberVars<Forcing_ConstDP>;
};

MIRHEO_MEMBER_VARS_1(Forcing_ConstDP, extraForce);

} // namespace mirheo
