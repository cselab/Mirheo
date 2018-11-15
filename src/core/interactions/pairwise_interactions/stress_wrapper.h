#pragma once

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>

class ParticleVector;
class CellList;

struct Stress
{
    float xx, xy, xz, yy, yz, zz;
};

template<typename BasicPairwiseForce>
class PairwiseStressWrapper
{
public:
    PairwiseStressWrapper(BasicPairwiseForce basicForce) : basicForce(basicForce)
    {    }

    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
    {
        basicForce.setup(lpv1, lpv2, cl1, cl2, t);

        pv1Stress = lpv1->extraPerParticle.getData<Stress>("stress")->devPtr();
        pv2Stress = lpv2->extraPerParticle.getData<Stress>("stress")->devPtr();
    }

    __device__ inline float3 operator()(const Particle dst, int dstId, const Particle src, int srcId) const
    {
        const float3 dr = dst.r - src.r;
        float3 f = basicForce(dst, dstId, src, srcId);

        const float sxx = 0.5f * dr.x * f.x;
        const float sxy = 0.5f * dr.x * f.y;
        const float sxz = 0.5f * dr.x * f.z;
        const float syy = 0.5f * dr.y * f.y;
        const float syz = 0.5f * dr.y * f.z;
        const float szz = 0.5f * dr.z * f.z;

        atomicAdd(&pv1Stress[dstId].xx, sxx);
        atomicAdd(&pv1Stress[dstId].xy, sxy);
        atomicAdd(&pv1Stress[dstId].xz, sxz);
        atomicAdd(&pv1Stress[dstId].yy, syy);
        atomicAdd(&pv1Stress[dstId].yz, syz);
        atomicAdd(&pv1Stress[dstId].zz, szz);

        atomicAdd(&pv2Stress[srcId].xx, sxx);
        atomicAdd(&pv2Stress[srcId].xy, sxy);
        atomicAdd(&pv2Stress[srcId].xz, sxz);
        atomicAdd(&pv2Stress[srcId].yy, syy);
        atomicAdd(&pv2Stress[srcId].yz, syz);
        atomicAdd(&pv2Stress[srcId].zz, szz);

        return f;
    }

private:

    Stress *pv1Stress, *pv2Stress;

    BasicPairwiseForce basicForce;
};
