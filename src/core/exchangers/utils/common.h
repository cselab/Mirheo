#pragma once

#include <core/pvs/packers/rods.h>

#include <extern/variant/include/mpark/variant.hpp>

using VarPackHandler = mpark::variant<ObjectPackerHandler, RodPackerHandler>;

namespace ExchangersCommon
{

__device__ inline int3 getDirection(float3 pos, float3 L)
{
    int3 dir {0, 0, 0};
    if (pos.x < -0.5f * L.x) dir.x = -1;
    if (pos.y < -0.5f * L.y) dir.y = -1;
    if (pos.z < -0.5f * L.z) dir.z = -1;

    if (pos.x >= 0.5f * L.x) dir.x = 1;
    if (pos.y >= 0.5f * L.y) dir.y = 1;
    if (pos.z >= 0.5f * L.z) dir.z = 1;
    
    return dir;
}

__device__ inline float3 getShift(float3 L, int3 dir)
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

} // namespace ExchangersCommon
