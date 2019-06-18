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

    __HD__ inline float3 normal(float3 r) const
    {
        return normalize(make_float3(
            axes.y*axes.y * axes.z*axes.z * r.x,
            axes.z*axes.z * axes.x*axes.x * r.y,
            axes.x*axes.x * axes.y*axes.y * r.z));
    }
    
    inline float3 inertiaTensor(float totalMass) const
    {
        return totalMass / 5.0f * make_float3
            (sqr(axes.y) + sqr(axes.z),
             sqr(axes.x) + sqr(axes.z),
             sqr(axes.x) + sqr(axes.y));
    }

    static const char *desc;
    
private:    
    float3 axes, invAxes;
};
