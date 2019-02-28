#pragma once

#include "real.h"

#include <core/mesh/membrane.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>

__D__ inline real triangleArea(real3 v0, real3 v1, real3 v2)
{
    return 0.5_r * length(cross(v1 - v0, v2 - v0));
}

__D__ inline real triangleSignedVolume(real3 v0, real3 v1, real3 v2)
{
    return 0.1666666667_r *
        (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
         - v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
}

__D__ inline real supplementaryDihedralAngle(real3 v0, real3 v1, real3 v2, real3 v3)
{
    //       v3
    //     /   \
    //   v2 --- v0
    //     \   /
    //       V
    //       v1

    // dihedral: 0123    

    real3 n, k, nk;
    n  = cross(v1 - v0, v2 - v0);
    k  = cross(v2 - v0, v3 - v0);
    nk = cross(n, k);

    real theta = atan2(length(nk), dot(n, k));
    theta = dot(v2-v0, nk) < 0 ? theta : -theta;
    return theta;
}
