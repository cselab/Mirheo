#pragma once

#include <utils/cpu_gpu_defines.h>
#include <utils/helper_math.h>

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

        float dist2edge = sqrtf(sq(dz) + sq(dr));
        float dist2disk = dr > 0 ? dist2edge : dz;
        float dist2cyl  = dz > 0 ? dist2edge : dr;

        return (dz <= 0 && dr <= 0)
                     ? max(dist2cyl, dist2disk)
                     : min(dist2cyl, dist2disk);
    }
    
private:
    
    float R, halfL;
};
