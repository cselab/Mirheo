#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/pvs/views/pv_with_stresses.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

struct ForceStress
{
    real3 force;
    Stress stress;
};

template <typename BasicView>
class ForceStressAccumulator
{
public:

    __D__ inline ForceStressAccumulator() :
        frcStress_({{0._r, 0._r, 0._r},
                    {0._r, 0._r, 0._r, 0._r, 0._r, 0._r}})
    {}
    
    __D__ inline void atomicAddToDst(const ForceStress& fs, PVviewWithStresses<BasicView>& view, int id) const
    {
        atomicAdd(      view.forces   + id, fs.force );
        atomicAddStress(view.stresses + id, fs.stress);
    }

    __D__ inline void atomicAddToSrc(const ForceStress& fs, PVviewWithStresses<BasicView>& view, int id) const
    {
        atomicAdd(      view.forces   + id, -fs.force );
        atomicAddStress(view.stresses + id,  fs.stress);
    }

    __D__ inline ForceStress get() const {return frcStress_;}

    __D__ inline void add(const ForceStress& fs)
    {
        frcStress_.force += fs.force;
        frcStress_.stress.xx += fs.stress.xx;
        frcStress_.stress.xy += fs.stress.xy;
        frcStress_.stress.xz += fs.stress.xz;
        frcStress_.stress.yy += fs.stress.yy;
        frcStress_.stress.yz += fs.stress.yz;
        frcStress_.stress.zz += fs.stress.zz;
    }
    
private:
    ForceStress frcStress_;

    __D__ inline void atomicAddStress(Stress *dst, const Stress& s) const
    {
        atomicAdd(&dst->xx, s.xx);
        atomicAdd(&dst->xy, s.xy);
        atomicAdd(&dst->xz, s.xz);
        atomicAdd(&dst->yy, s.yy);
        atomicAdd(&dst->yz, s.yz);
        atomicAdd(&dst->zz, s.zz);
    }
};

} // namespace mirheo
