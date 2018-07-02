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
	{	}

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

	    const float v0 = 0.5f * dr.x*f.x;
	    const float v1 = 0.5f * dr.x*f.y;
	    const float v2 = 0.5f * dr.x*f.z;
	    const float v3 = 0.5f * dr.y*f.y;
	    const float v4 = 0.5f * dr.y*f.z;
	    const float v5 = 0.5f * dr.z*f.z;

	    atomicAdd(&pv1Stress[dstId].xx, v0);
	    atomicAdd(&pv1Stress[dstId].xy, v1);
	    atomicAdd(&pv1Stress[dstId].xz, v2);
	    atomicAdd(&pv1Stress[dstId].yy, v3);
	    atomicAdd(&pv1Stress[dstId].yz, v4);
	    atomicAdd(&pv1Stress[dstId].zz, v5);

	    atomicAdd(&pv2Stress[srcId].xx, v0);
	    atomicAdd(&pv2Stress[srcId].xy, v1);
	    atomicAdd(&pv2Stress[srcId].xz, v2);
	    atomicAdd(&pv2Stress[srcId].yy, v3);
	    atomicAdd(&pv2Stress[srcId].yz, v4);
	    atomicAdd(&pv2Stress[srcId].zz, v5);

	    return f;
	}

private:

	Stress *pv1Stress, *pv2Stress;

	BasicPairwiseForce basicForce;
};
