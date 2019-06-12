#pragma once

#include "../utils/fragments_mapping.h"

#include <core/domain.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/type_map.h>
#include <core/utils/type_shift.h>

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
        
            TypeShift::shift(var, shift);
        }
        return var;
    }

private:
    const bool needShift;
    const DomainInfo domain;
};
