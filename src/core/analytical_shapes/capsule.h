#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

class Capsule
{
public:
    Capsule(float R, float L) :
        R(R),
        halfL(0.5 * L)
    {}

    __HD__ inline float inOutFunction(float3 coo) const
    {
        float dz = fabs(coo.z) - halfL;

        float drsq = sqr(coo.x) + sqr(coo.y);
        if (dz > 0) drsq += sqr(dz);

        float dr = sqrtf(drsq) - R;
        return dr;
    }

    __HD__ inline float3 normal(float3 coo) const
    {
        constexpr float eps = 1e-6f;

        float dz = fabs(coo.z) - halfL;

        float rsq = sqr(coo.x) + sqr(coo.y);
        if (dz > 0) rsq += sqr(dz);

        float rinv = rsq > eps ? rsqrtf(rsq) : 0.f;

        float3 n {coo.x,
                  coo.y,
                  dz > 0 ? dz : 0.f};
        return rinv * n;
    }
    

    inline float3 inertiaTensor(float totalMass) const
    {
        const float R2 = R * R;
        const float R3 = R2 * R;
        const float R4 = R2 * R2;
        const float R5 = R3 * R2;
        
        const float V_pi   = 2 * halfL * R2 + 4.0 / 3.0 * R3;
        
        const float xxB_pi = R5 * 4.0 / 15.0;
        const float xxC_pi = R4 * halfL / 2.0;

        const float zzB_pi = 4.0 * (halfL * halfL * R3 / 3.0
                                    + halfL * R4 / 4.0
                                    + R5 / 15.0);
        const float zzC_pi = R2 * halfL * halfL * halfL * 2.0 / 3.0;

        const float xx = totalMass * (xxB_pi + xxC_pi) / V_pi;
        const float zz = totalMass * (zzB_pi + zzC_pi) / V_pi;
        const float yy = xx;
                
        return make_float3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;
    
private:
    
    float R, halfL;
};
