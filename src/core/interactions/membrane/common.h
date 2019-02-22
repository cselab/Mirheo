#pragma once

#include <core/mesh/membrane.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>

template <typename View>
__D__ inline float3 fetchPosition(View view, int i)
{
    Particle p;
    p.readCoordinate(view.particles, i);
    return p.r;
}

__D__ inline float triangleArea(float3 v0, float3 v1, float3 v2)
{
    return 0.5f * length(cross(v1 - v0, v2 - v0));
}

__D__ inline float triangleSignedVolume(float3 v0, float3 v1, float3 v2)
{
    return 0.1666666667f *
        (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
         - v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
}

__D__ inline float supplementaryDihedralAngle(float3 v0, float3 v1, float3 v2, float3 v3)
{
    //       v3
    //     /   \
    //   v2 --- v0
    //     \   /
    //       V
    //       v1

    // dihedral: 0123    

    float3 n, k, nk;
    n  = cross(v1 - v0, v2 - v0);
    k  = cross(v2 - v0, v3 - v0);
    nk = cross(n, k);

    float theta = atan2(length(nk), dot(n, k));
    theta = dot(v2-v0, nk) < 0 ? theta : -theta;
    return theta;
}
