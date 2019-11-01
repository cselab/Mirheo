#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class Ellipsoid
{
public:
    Ellipsoid(real3 axes) :
        axes(axes),
        invAxes(1.0 / axes)
    {}

    __HD__ inline real inOutFunction(real3 r) const
    {
        return sqr(r.x * invAxes.x) + sqr(r.y * invAxes.y) + sqr(r.z * invAxes.z) - 1.0_r;
    }

    __HD__ inline real3 normal(real3 r) const
    {
        return normalize(make_real3(
            axes.y*axes.y * axes.z*axes.z * r.x,
            axes.z*axes.z * axes.x*axes.x * r.y,
            axes.x*axes.x * axes.y*axes.y * r.z));
    }
    
    inline real3 inertiaTensor(real totalMass) const
    {
        return totalMass / 5.0_r * make_real3
            (sqr(axes.y) + sqr(axes.z),
             sqr(axes.x) + sqr(axes.z),
             sqr(axes.x) + sqr(axes.y));
    }

    static const char *desc;
    
private:    
    real3 axes, invAxes;
};

} // namespace mirheo
