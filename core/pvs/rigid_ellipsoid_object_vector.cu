#include "rigid_ellipsoid_object_vector.h"

#include <core/rigid_kernels/quaternion.h>
#include <core/cuda_common.h>

__device__ __forceinline__ bool inside(float3 r, LocalRigidObjectVector::RigidMotion motion, float3 invAxes)
{
	float3 r_loc = rotate(r - motion.r, motion.q);
	return ( sqr(r.x*invAxes.x) + sqr(r.y*invAxes.y) + sqr(r.z*invAxes.z) - 1.0f ) < 0.0f;
}

__global__ void computeInsideOutsideTags(
		const float4* coosvels, int np,
		const LocalRigidObjectVector::RigidMotion* __restrict__ motions, float3 invAxes, int nObjs,
		int* tag, int* nInside, int* nOutside)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	if (pid >= np) return;

	const float3 r = f4tof3(coosvels[2*pid]);

	bool in = false;
	for (int objId=0; objId<nObjs; objId++)
		if (inside(r, motions[objId], invAxes))
		{
			in = true;
			break;
		}

	if (in)
	{
		atomicAggInc(nInside);
		tag[pid] = 1;
	}
	else
	{
		atomicAggInc(nOutside);
		tag[pid] = 0;
	}
}

void RigidEllipsoidObjectVector::inside(ParticleVector* pv, PinnedBuffer<int>& tags, int& totalInside, int& totalOutsize, cudaStream_t stream)
{
	PinnedBuffer<int> nInside(1), nOutside(1);
	tags.resize(pv->local()->size(), stream, resizeAnew);
	nInside.clear(stream);
	nOutside.clear(stream);

	const int nthreads = 128;

	computeInsideOutsideTags <<< getNblocks(pv->local()->size(), nthreads), nthreads, 0, stream >>>(
			(float4*)pv->local()->coosvels.devPtr(), pv->local()->size(),
			local()->motions.devPtr(), 1.0 / axes, local()->nObjects,
			tags.devPtr(), nInside.devPtr(), nOutside.devPtr());

	nInside. downloadFromDevice(stream);
	nOutside.downloadFromDevice(stream);

	totalInside  = nInside[0];
	totalOutsize = nOutside[0];
};

Particle* RigidEllipsoidObjectVector::getMeshVertices()
{
	return nullptr;
}

float3 RigidEllipsoidObjectVector::getInertiaTensor()
{
	return objMass / 5.0 * make_float3(
			sqr(axes.y) + sqr(axes.z),
			sqr(axes.z) + sqr(axes.x),
			sqr(axes.x) + sqr(axes.y) );
}




