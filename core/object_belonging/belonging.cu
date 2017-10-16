#include "interface.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

__global__ void copyLeftRight(
		PVview view,
		const int* tags,
		Particle* ins, Particle* outs,
		int* nIn, int* nOut)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= view.size) return;

	const int tag = tags[gid];
	const Particle p(view.particles, gid);

	if (tag == -1)
	{
		int dstId = atomicAggInc(nIn);
		if (ins)  ins [dstId] = p;
	}

	if (tag >= 0)
	{
		int dstId = atomicAggInc(nOut);
		if (outs) outs[dstId] = p;
	}
}


static void ObjectBelongingChecker::splitByTags(ParticleVector* src, CellList* cl, ParticleVector* pvIn, ParticleVector* pvOut, cudaStream_t stream)
{
	if (dynamic_cast<ObjectVector*>(src) != nullptr)
		error("Trying to split object vector %s into two per-particle, probably that's not what you wanted",
				src->name.c_str());

	if (pvIn != nullptr && typeid(*src) != typeid(*pvIn))
		error("PV type of inner result of split (%s) is different from source (%s)",
				pvIn->name.c_str(), src->name.c_str());

	if (pvOut != nullptr && typeid(*src) != typeid(*pvOut))
		error("PV type of outer result of split (%s) is different from source (%s)",
				pvOut->name.c_str(), src->name.c_str());

	tagInner(pv, cl, stream);

	if (pvIn != nullptr)  pvIn-> local()->resize(nInside,  stream);
	if (pvOut != nullptr) pvOut->local()->resize(nOutside, stream);

	nInside. clearDevice(stream);
	nOutside.clearDevice(stream);

	SAFE_KERNEL_LAUNCH(
			copyLeftRight,
			getNblocks(src->local()->size(), 128), 128, 0, stream,
			src->local()->coosvels.devPtr(), src->local()->size(),
			tags.devPtr(),
			pvIn ?  pvIn-> local()->coosvels.devPtr() : nullptr,
			pvOut ? pvOut->local()->coosvels.devPtr() : nullptr,
			nInside.devPtr(), nOutside.devPtr() );
}
