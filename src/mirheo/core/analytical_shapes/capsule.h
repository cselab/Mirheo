#pragma once

#include "interface.h"

#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class Capsule: public AnalyticShape
{
public:
    Capsule(real R, real L) :
        R_(R),
        halfL_(0.5_r * L)
    {}

    __HD__ real inOutFunction(real3 coo) const override
    {
        const real dz = math::abs(coo.z) - halfL_;

        real drsq = sqr(coo.x) + sqr(coo.y);
        if (dz > 0) drsq += sqr(dz);

        const real dr = math::sqrt(drsq) - R_;
        return dr;
    }

    __HD__ real3 normal(real3 coo) const override
    {
        constexpr real eps = 1e-6_r;

        const real dz = math::abs(coo.z) - halfL_;

        real rsq = sqr(coo.x) + sqr(coo.y);
        if (dz > 0) rsq += sqr(dz);

        const real rinv = rsq > eps ? math::rsqrt(rsq) : 0._r;

        const real3 n {coo.x,
                        coo.y,
                        dz > 0 ? dz : 0._r};
        return rinv * n;
    }
    

    real3 inertiaTensor(real totalMass) const override
    {
        const real R2 = R_ * R_;
        const real R3 = R2 * R_;
        const real R4 = R2 * R2;
        const real R5 = R3 * R2;
        
        const real V_pi   = 2.0_r * halfL_ * R2 + (4.0_r / 3.0_r) * R3;
        
        const real xxB_pi = R5 * (4.0_r / 15.0_r);
        const real xxC_pi = R4 * halfL_ * 0.5_r;

        const real zzB_pi = 4.0_r * (halfL_ * halfL_ * R3 / 3.0_r
                                     + halfL_ * R4 / 4.0_r
                                     + R5 / 15.0_r);
        const real zzC_pi = R2 * halfL_ * halfL_ * halfL_ * (2.0_r / 3.0f);

        const real xx = totalMass * (xxB_pi + xxC_pi) / V_pi;
        const real zz = totalMass * (zzB_pi + zzC_pi) / V_pi;
        const real yy = xx;

        return make_real3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;
    
private:
    
    real R_, halfL_;
};

} // namespace mirheo
