#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

class Cylinder
{
public:
    Cylinder(float R, float L) :
        R(R),
        halfL(0.5f * L)
    {}

    __HD__ inline float inOutFunction(float3 coo) const
    {
        const float dr = sqrtf(sqr(coo.x) + sqr(coo.y)) - R;
        const float dz = fabsf(coo.z) - halfL;

        const float dist2edge = sqrtf(sqr(dz) + sqr(dr));
        const float dist2disk = dr > 0 ? dist2edge : dz;
        const float dist2cyl  = dz > 0 ? dist2edge : dr;

        return (dz <= 0) && (dr <= 0)
            ? fmaxf(dist2cyl, dist2disk)
            : fminf(dist2cyl, dist2disk);
    }

    __HD__ inline float3 normal(float3 coo) const
    {
        constexpr float eps   = 1e-6f;
        constexpr float delta = 1e-3f;
        
        const float rsq = sqr(coo.x) + sqr(coo.y);
        const float rinv = rsq > eps ? rsqrtf(rsq) : 0.f;

        const float dr = sqrtf(rsq) - R;
        const float dz = fabsf(coo.z) - halfL;
        
        const float3 er {rinv * coo.x, rinv * coo.y, 0.f};
        const float3 ez {0.f, 0.f, coo.z > 0 ? 1.f : -1.f};

        
        float3 n {0.f, 0.f, 0.f};
        if (fabsf(dr) < delta) n += er;
        if (fabsf(dz) < delta) n += ez;
        return n;
    }
    

    inline float3 inertiaTensor(float totalMass) const
    {
        const float xx = totalMass * R * R * 0.25f;
        const float yy = xx;
        const float zz = totalMass * halfL * halfL * 0.3333333f;
        
        return make_float3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;
    
private:
    
    float R, halfL;
};
