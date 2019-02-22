#pragma once

#include "../parameters.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <cmath>

class TriangleForce
{
public:    

    TriangleForce(float ka, float kd, float totArea0, float lscale) :
        ka(ka * lscale * lscale),
        kd(kd * lscale * lscale),
        totArea0(totArea0 / (lscale*lscale))
    {}

    __D__ inline float3 bondForce(VertexType v0, VertexType v1, float l0) const
    {
        return make_float3(0.f);
    }

    __D__ inline float3 areaForce(const float3 v1, const float3 v2, const float3 v3,
                                  const float area0, const float totArea) const
    {
        float3 x21 = v2 - v1;
        float3 x32 = v3 - v2;
        float3 x31 = v3 - v1;

        float3 normal = cross(x21, x31);

        float area = 0.5f * length(normal);
        float inv_area = 1.0f / area;

        float coefArea = ka * (totArea - totArea0) * inv_area
                       + kd * (area - area0) / (area * area0);

        return -0.25f * coefArea * cross(normal, x32);
    }

protected:
    
    float ka, kd, totArea0;
};
