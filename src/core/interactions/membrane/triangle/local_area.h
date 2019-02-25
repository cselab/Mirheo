#pragma once

#include "../parameters.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <cmath>

class LocalAreaForce
{
public:    

    LocalAreaForce(float kd, float lscale) :
        kd (kd * lscale * lscale)
    {}

    __D__ inline float3 operator()(float3 v1, float3 v2, float3 v3, float l0, float a0) const
    {
        return areaForce(v1, v2, v3, a0);
    }

protected:

    __D__ inline float3 areaForce(float3 v1, float3 v2, float3 v3, float area0) const
    {
        float3 x21 = v2 - v1;
        float3 x32 = v3 - v2;
        float3 x31 = v3 - v1;

        float3 normal = cross(x21, x31);

        float area = 0.5f * length(normal);

        float coef = kd * (area - area0) / (area * area0);

        return -0.25f * coef * cross(normal, x32);
    }

    float kd;
};
