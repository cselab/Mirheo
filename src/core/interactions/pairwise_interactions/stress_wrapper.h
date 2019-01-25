#pragma once

#include <core/datatypes.h>
#include <core/interactions/accumulators/forceStress.h>
#include <core/utils/common.h>

class LocalParticleVector;
class CellList;

template<typename BasicPairwiseForce>
class PairwiseStressWrapper
{
public:

    using ViewType = PVviewWithStresses;
    
    PairwiseStressWrapper(BasicPairwiseForce basicForce) :
        basicForce(basicForce)
    {}

    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
    {
        basicForce.setup(lpv1, lpv2, cl1, cl2, t);
    }

    __device__ inline ForceStress operator()(const Particle dst, int dstId, const Particle src, int srcId) const
    {        
        float3 dr = dst.r - src.r;
        float3 f = basicForce(dst, dstId, src, srcId);
        Stress s;
        
        s.xx = 0.5f * dr.x * f.x;
        s.xy = 0.5f * dr.x * f.y;
        s.xz = 0.5f * dr.x * f.z;
        s.yy = 0.5f * dr.y * f.y;
        s.yz = 0.5f * dr.y * f.z;
        s.zz = 0.5f * dr.z * f.z;        

        return {f, s};
    }

    __D__ inline ForceStressAccumulator getZeroedAccumulator() const {return ForceStressAccumulator();}

private:
    
    BasicPairwiseForce basicForce;
};
