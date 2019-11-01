#pragma once

#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

class DensityAccumulator
{
public:

    __D__ inline DensityAccumulator() :
        den(0._r)
    {}
    
    __D__ inline void atomicAddToDst(real d, PVviewWithDensities& view, int id) const
    {
        atomicAdd(view.densities + id, d);
    }

    __D__ inline void atomicAddToSrc(real d, PVviewWithDensities& view, int id) const
    {
        atomicAdd(view.densities + id, d);
    }

    __D__ inline real get() const {return den;}
    __D__ inline void add(real d) {den += d;}
    
private:
    real den;
};
