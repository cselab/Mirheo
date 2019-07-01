#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

class Cylinder
{
public:
    Cylinder(float R, float L) :
        R(R),
        halfL(0.5 * L)
    {}

    __HD__ inline float inOutFunction(float3 coo) const
    {
        float dr = sqrtf(sqr(coo.x) + sqr(coo.y)) - R;
        float dz = fabs(coo.z) - halfL;

        float dist2edge = sqrtf(sqr(dz) + sqr(dr));
        float dist2disk = dr > 0 ? dist2edge : dz;
        float dist2cyl  = dz > 0 ? dist2edge : dr;

        return (dz <= 0) && (dr <= 0)
            ? fmax(dist2cyl, dist2disk)
            : fmin(dist2cyl, dist2disk);
    }

    __HD__ inline float3 normal(float3 coo) const
    {
        constexpr float eps   = 1e-6f;
        constexpr float delta = 1e-3f;
        
        float rsq = sqr(coo.x) + sqr(coo.y);
        float rinv = rsq > eps ? rsqrtf(rsq) : 0.f;

        float dr = sqrtf(rsq) - R;
        float dz = fabs(coo.z) - halfL;
        
        float3 er {rinv * coo.x, rinv * coo.y, 0.f};
        float3 ez {0.f, 0.f, coo.z > 0 ? 1.f : -1.f};

        
        float3 n {0.f, 0.f, 0.f};
        if (fabs(dr) < delta) n += er;
        if (fabs(dz) < delta) n += ez;
        return n;
    }
    

    inline float3 inertiaTensor(float totalMass) const
    {
        const float xx = totalMass * R * R / 4.0;
        const float yy = xx;
        const float zz = totalMass * halfL * halfL / 3.0;
        
        return make_float3(yy + zz, xx + zz, xx + yy);
    }

    static const char *desc;
    
private:
    
    float R, halfL;
};
