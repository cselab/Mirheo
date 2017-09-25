#include "object_forces_reverse_exchanger.h"

#include "object_halo_exchanger.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/logger.h>
#include <core/cuda_common.h>


// TODO: change id scheme
__global__ void addHaloForces(const float4* haloForces, const float4* halo, float4* forces, int objSize, int n)
{
	const int srcId = blockIdx.x*blockDim.x + threadIdx.x;
	if (srcId >= n) return;

	const Particle p(halo[2*srcId], halo[2*srcId+1]);
	const int dstId = p.s22 /* objId */ * objSize + p.s21 /* pid in object */;

	const Float3_int extraFrc = readNoCache(haloForces + srcId);
	Float3_int frc0 = forces[dstId];
	frc0.v += extraFrc.v;

	forces[dstId] = frc0.toFloat4();
}


void ObjectForcesReverseExchanger::attach(ObjectVector* ov)
{
	objects.push_back(ov);
	ExchangeHelper* helper = new ExchangeHelper(ov->name, ov->local()->objSize*sizeof(Force));
	helpers.push_back(helper);
}


void ObjectForcesReverseExchanger::prepareData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];
	auto offsets = entangledHaloExchanger->helpers[id]->recvOffsets;

	debug2("Preparing %s forces to sending back", ov->name.c_str());

	for (int i=0; i<27; i++)
		helper->sendBufSizes[i] = offsets[i+1] - offsets[i];
	helper->resizeSendBufs();

	for (int i=0; i<27; i++)
	{
		if (helper->sendBufSizes[i] > 0)
			CUDA_Check( cudaMemcpyAsync( helper->sendBufs[i].hostPtr(),
										 ov->halo()->forces.devPtr() + offsets[i]*ov->halo()->objSize,
										 helper->sendBufSizes[i]*sizeof(Force)*ov->halo()->objSize,
										 cudaMemcpyHostToDevice, stream ) );
	}
}

void ObjectForcesReverseExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];

	for (int i=0; i < helper->recvOffsets.size() - 1; i++)
	{
		const int msize = helper->recvOffsets[i+1] - helper->recvOffsets[i];

		debug3("Updating forces for %d %s objects", msize, ov->name.c_str());

		if (msize > 0)
			CUDA_Check( cudaMemcpyAsync(ov->halo()->forces.devPtr() + helper->recvOffsets[i]*ov->halo()->objSize,
										helper->recvBufs[compactedDirs[i]].hostPtr(),
										msize*sizeof(Force)*ov->halo()->objSize,
										cudaMemcpyHostToDevice, stream) );
	}

	const int np = helper->recvOffsets[27];
	if (np > 0)
		addHaloForces<<< (np+127)/128, 128, 0, stream >>> (
				(float4*)ov->halo()->forces.devPtr(),    /* add to */
				(float4*)ov->halo()->coosvels.devPtr(),  /* destination id here */
				(float4*)ov->local()->forces.devPtr(),   /* source */
				ov->objSize, np );
}





