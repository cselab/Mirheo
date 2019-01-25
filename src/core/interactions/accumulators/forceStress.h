#pragma once

#include <core/datatypes.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

struct ForceStress
{
    float3 force;
    Stress stress;
};

class ForceStressAccumulator
{
public:

    __D__ inline ForceStressAccumulator() :
        frcStress({{0.f, 0.f, 0.f},
                   {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}})
    {}
    
    __D__ inline void atomicAddToDst(const ForceStress& fs, PVviewWithStresses& view, int id) const
    {
        atomicAdd(view.forces   + id, fs.force );
        atomicAddStress(view.stresses + id, fs.stress);
    }

    __D__ inline void atomicAddToSrc(const ForceStress& fs, PVviewWithStresses& view, int id) const
    {
        atomicAdd(view.forces   + id, -fs.force );
        atomicAddStress(view.stresses + id,  fs.stress);
    }

    __D__ inline ForceStress get() const {return frcStress;}

    __D__ inline void add(const ForceStress& fs)
    {
        frcStress.force += fs.force;
        frcStress.stress.xx += fs.stress.xx;
        frcStress.stress.xy += fs.stress.xy;
        frcStress.stress.xz += fs.stress.xz;
        frcStress.stress.yy += fs.stress.yy;
        frcStress.stress.yz += fs.stress.yz;
        frcStress.stress.zz += fs.stress.zz;
    }
    
private:
    ForceStress frcStress;

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
