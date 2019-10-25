#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

class Capsule
{
public:
    Capsule(real R, real L) :
        R(R),
        halfL(0.5f * L)
    {}

    __HD__ inline real inOutFunction(real3 coo) const
    {
        const real dz = math::abs(coo.z) - halfL;

        real drsq = sqr(coo.x) + sqr(coo.y);
        if (dz > 0) drsq += sqr(dz);

        const real dr = math::sqrt(drsq) - R;
        return dr;
    }

    __HD__ inline real3 normal(real3 coo) const
    {
        constexpr real eps = 1e-6f;

        const real dz = math::abs(coo.z) - halfL;

        real rsq = sqr(coo.x) + sqr(coo.y);
        if (dz > 0) rsq += sqr(dz);

        const real rinv = rsq > eps ? math::rsqrt(rsq) : 0.f;

        const real3 n {coo.x,
                        coo.y,
                        dz > 0 ? dz : 0.f};
        return rinv * n;
    }
    

    inline real3 inertiaTensor(real totalMass) const
    {
        const real R2 = R * R;
        const real R3 = R2 * R;
        const real R4 = R2 * R2;
        const real R5 = R3 * R2;
        
        const real V_pi   = 2.0f * halfL * R2 + (4.0f / 3.0f) * R3;
        
        const real xxB_pi = R5 * (4.0f / 15.0f);
        const real xxC_pi = R4 * halfL * 0.5f;

        const real zzB_pi = 4.0f * (halfL * halfL * R3 / 3.0f
                                    + halfL * R4 / 4.0f
                                    + R5 / 15.0f);
        const real zzC_pi = R2 * halfL * halfL * halfL * (2.0f / 3.0f);

        const real xx = totalMass * (xxB_pi + xxC_pi) / V_pi;
        const real zz = totalMass * (zzB_pi + zzC_pi) / V_pi;
        const real yy = xx;

        return make_real3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;
    
private:
    
    real R, halfL;
};
