#pragma once

#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

namespace mirheo
{

class DensityAccumulator
{
public:

    __D__ inline DensityAccumulator() :
        den_(0._r)
    {}
    
    __D__ inline void atomicAddToDst(real d, PVviewWithDensities& view, int id) const
    {
        atomicAdd(view.densities + id, d);
    }

    __D__ inline void atomicAddToSrc(real d, PVviewWithDensities& view, int id) const
    {
        atomicAdd(view.densities + id, d);
    }

    __D__ inline real get() const {return den_;}
    __D__ inline void add(real d) {den_ += d;}
    
private:
    real den_;
};

} // namespace mirheo
