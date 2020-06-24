// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/pvs/packers/rods.h>

#include <extern/variant/include/mpark/variant.hpp>

namespace mirheo
{

/// generic object packer
using VarPackHandler = mpark::variant<ObjectPackerHandler, RodPackerHandler>;

namespace exchangers_common
{

__device__ static inline int3 getDirection(real3 pos, real3 L)
{
    int3 dir {0, 0, 0};
    if (pos.x < -0.5_r * L.x) dir.x = -1;
    if (pos.y < -0.5_r * L.y) dir.y = -1;
    if (pos.z < -0.5_r * L.z) dir.z = -1;

    if (pos.x >= 0.5_r * L.x) dir.x = 1;
    if (pos.y >= 0.5_r * L.y) dir.y = 1;
    if (pos.z >= 0.5_r * L.z) dir.z = 1;

    return dir;
}

__device__ static inline real3 getShift(real3 L, int3 dir)
{
    return {-L.x * dir.x,
            -L.y * dir.y,
            -L.z * dir.z};
}

inline VarPackHandler getHandler(ObjectPacker *packer)
{
    auto rod = dynamic_cast<RodPacker*>(packer);

    if (rod) return rod->handler();
    return packer->handler();
}

} // namespace exchangers_common

} // namespace mirheo
