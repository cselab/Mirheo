#pragma once

#include "../utils/fragments_mapping.h"

#include <core/domain.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/type_map.h>

struct Shifter
{
    Shifter(bool needShift, DomainInfo domain) :
        needShift(needShift),
        domain(domain)
    {}

    template <typename T>
    __D__ inline T operator()(T var, int bufId) const
    {
        if (needShift)
        {
            int3 dir = FragmentMapping::getDir(bufId);
            float3 shift { -domain.localSize.x * dir.x,
                           -domain.localSize.y * dir.y,
                           -domain.localSize.z * dir.z };
        
            _shift(var, shift);
        }
        return var;
    }

private:

    template <typename T>
    __D__ inline void _shift(T& var, float3 shift) const {}

    __D__ inline void _shift(float3&      var, float3 shift) const {_add(var,   shift);}
    __D__ inline void _shift(float4&      var, float3 shift) const {_add(var,   shift);}
    __D__ inline void _shift(double3&     var, float3 shift) const {_add(var,   shift);}
    __D__ inline void _shift(double4&     var, float3 shift) const {_add(var,   shift);}
    __D__ inline void _shift(RigidMotion& var, float3 shift) const {_add(var.r, shift);}

    __D__ inline void _shift(COMandExtent& var, float3 shift) const
    {
        _add(var.com,  shift);
        _add(var.low,  shift);
        _add(var.high, shift);
    }

    template <typename T>
    __D__ inline void _add(T& v, float3 s) const
    {
        v.x += s.x;
        v.y += s.y;
        v.z += s.z;
    }
    
    const bool needShift;
    const DomainInfo domain;
};
