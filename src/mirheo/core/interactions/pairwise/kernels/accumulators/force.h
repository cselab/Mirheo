#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

class ForceAccumulator
{
public:

    __D__ inline ForceAccumulator() :
        frc({0._r, 0._r, 0._r})
    {}
    
    __D__ inline void atomicAddToDst(real3 f, PVview& view, int id) const
    {
        atomicAdd(view.forces + id, f);
    }

    __D__ inline void atomicAddToSrc(real3 f, PVview& view, int id) const
    {
        atomicAdd(view.forces + id, -f);
    }

    __D__ inline real3 get() const {return frc;}
    __D__ inline void add(real3 f) {frc += f;}
    
private:
    real3 frc;
};

} // namespace mirheo
