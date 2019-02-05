#pragma once

#include <core/datatypes.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

class ForceAccumulator
{
public:

    __D__ inline ForceAccumulator() :
        frc({0.f, 0.f, 0.f})
    {}
    
    __D__ inline void atomicAddToDst(float3 f, PVview& view, int id) const
    {
        atomicAdd(view.forces + id, f);
    }

    __D__ inline void atomicAddToSrc(float3 f, PVview& view, int id) const
    {
        atomicAdd(view.forces + id, -f);
    }

    __D__ inline float3 get() const {return frc;}
    __D__ inline void add(float3 f) {frc += f;}
    
private:
    float3 frc;
};
