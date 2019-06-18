#pragma once

#include <utils/cpu_gpu_defines.h>
#include <utils/helper_math.h>

class Ellipsoid
{
public:
    Ellipsoid(float3 axes) :
        axes(axes),
        invAxes(1.0 / axes)
    {}

    __HD__ inline float inOutFunction(float3 r) const
    {
        return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1.0f;
    }
    
private:    
    float3 axes, invAxes;
};
