#pragma once

#include "real.h"

#include <mirheo/core/mesh/membrane.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>

namespace mirheo
{

__D__ inline mReal triangleArea(mReal3 v0, mReal3 v1, mReal3 v2)
{
    return 0.5_mr * length(cross(v1 - v0, v2 - v0));
}

__D__ inline mReal triangleSignedVolume(mReal3 v0, mReal3 v1, mReal3 v2)
{
    return 0.1666666667_mr *
        (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
         - v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
}

__D__ inline mReal supplementaryDihedralAngle(mReal3 v0, mReal3 v1, mReal3 v2, mReal3 v3)
{
    /*
           v3
         /   \
       v2 --- v0
         \   /
           V
           v1

     dihedral: 0123    
    */
    
    mReal3 n, k, nk;
    n  = cross(v1 - v0, v2 - v0);
    k  = cross(v2 - v0, v3 - v0);
    nk = cross(n, k);

    mReal theta = atan2(length(nk), dot(n, k));
    theta = dot(v2-v0, nk) < 0 ? theta : -theta;
    return theta;
}

} // namespace mirheo
