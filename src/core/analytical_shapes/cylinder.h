#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

class Cylinder
{
public:
    Cylinder(float R, float L) :
        R(R),
        halfL(0.5*L)
    {}

    __HD__ inline float inOutFunction(float3 x) const
    {
        float dr = sqrtf(sqr(x.x) + sqr(x.y)) - R;
        float dz = fabs(x.z) - halfL;

        float dist2edge = sqrtf(sqr(dz) + sqr(dr));
        float dist2disk = dr > 0 ? dist2edge : dz;
        float dist2cyl  = dz > 0 ? dist2edge : dr;

        return (dz <= 0 && dr <= 0)
                     ? max(dist2cyl, dist2disk)
                     : min(dist2cyl, dist2disk);
    }

    inline float3 inertiaTensor(float totalMass) const
    {
        const float xx = totalMass * R * R / 8.0;
        const float yy = xx;
        const float zz = totalMass * halfL * halfL / 6.0;
        
        return make_float3(yy + zz, xx + zz, xx + yy);
    }

private:
    
    float R, halfL;
};
