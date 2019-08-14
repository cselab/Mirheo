#pragma once

#include <core/domain.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/type_map.h>

namespace TypeShift
{
template <typename T>
__HD__ inline void _add(T& v, float3 s)
{
    v.x += s.x;
    v.y += s.y;
    v.z += s.z;
}

template <typename T>
__HD__ inline void apply(__UNUSED T& var, __UNUSED float3 shift) {}

__HD__ inline void apply(float3&      var, float3 shift) {_add(var,   shift);}
__HD__ inline void apply(float4&      var, float3 shift) {_add(var,   shift);}
__HD__ inline void apply(double3&     var, float3 shift) {_add(var,   shift);}
__HD__ inline void apply(double4&     var, float3 shift) {_add(var,   shift);}
__HD__ inline void apply(RigidMotion& var, float3 shift) {_add(var.r, shift);}

__HD__ inline void apply(COMandExtent& var, float3 shift)
{
    _add(var.com,  shift);
    _add(var.low,  shift);
    _add(var.high, shift);
}

} // namespace TypeShift
