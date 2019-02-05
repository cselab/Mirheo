#pragma once

#include <core/pvs/views/pv.h>
#include <core/utils/cpu_gpu_defines.h>

class DensityAccumulator
{
public:

    __D__ inline DensityAccumulator() :
        den(0.f)
    {}
    
    __D__ inline void atomicAddToDst(float d, PVviewWithDensities& view, int id) const
    {
        atomicAdd(view.densities + id, d);
    }

    __D__ inline void atomicAddToSrc(float d, PVviewWithDensities& view, int id) const
    {
        atomicAdd(view.densities + id, d);
    }

    __D__ inline float get() const {return den;}
    __D__ inline void add(float d) {den += d;}
    
private:
    float den;
};
