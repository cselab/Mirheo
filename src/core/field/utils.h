#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

template <typename FieldHandler>
inline __D__ float3 computeGradient(const FieldHandler& field, float3 x, float h)
{
    float mx = field(x + make_float3(-h,  0,  0));
    float px = field(x + make_float3( h,  0,  0));
    float my = field(x + make_float3( 0, -h,  0));
    float py = field(x + make_float3( 0,  h,  0));
    float mz = field(x + make_float3( 0,  0, -h));
    float pz = field(x + make_float3( 0,  0,  h));

    float3 diff { px - mx,
                  py - my,
                  pz - mz };

    return (1.0f / (2.0f*h)) * diff;
}
