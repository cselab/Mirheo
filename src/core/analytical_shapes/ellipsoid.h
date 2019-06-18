#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

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

    inline float3 inertiaTensor(float totalMass) const
    {
        return totalMass / 5.0f * make_float3
            (sqr(axes.y) + sqr(axes.z),
             sqr(axes.x) + sqr(axes.z),
             sqr(axes.x) + sqr(axes.y));
    }
    
private:    
    float3 axes, invAxes;
};
