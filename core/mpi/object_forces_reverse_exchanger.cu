#include "object_forces_reverse_exchanger.h"

#include "object_halo_exchanger.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>


__device__ __forceinline__ void atomicAdd(float4* dest, float3 v)
{
	float* fdest = (float*)dest;
	atomicAdd(fdest,     v.x);
	atomicAdd(fdest + 1, v.y);
	atomicAdd(fdest + 2, v.z);
}

// TODO: change id scheme
__global__ void addHaloForces(const float4** recvForces, const int** origins, float4* forces, int* bufSizes)
{
	const int bufId = blockIdx.y;
	const int srcId = blockIdx.x*blockDim.x + threadIdx.x;
	if (srcId >= bufSizes[bufId]) return;

	const int dstId = origins[bufId][srcId];

	const Float3_int extraFrc( recvForces[bufId][srcId] );

	atomicAdd(forces + dstId, extraFrc.v);
}


void ObjectForcesReverseExchanger::attach(ObjectVector* ov)
{
	objects.push_back(ov);
	ExchangeHelper* helper = new ExchangeHelper(ov->name, ov->objSize*sizeof(Force));
	helpers.push_back(helper);
}


void ObjectForcesReverseExchanger::prepareData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];
	auto& offsets = entangledHaloExchanger->getRecvOffsets(id);

	debug2("Preparing %s forces to sending back", ov->name.c_str());

	for (int i=0; i<27; i++)
		helper->sendBufSizes[i] = offsets[i+1] - offsets[i];
	helper->resizeSendBufs();

	for (int i=0; i<27; i++)
	{
		if (helper->sendBufSizes[i] > 0)
			CUDA_Check( cudaMemcpyAsync( helper->sendBufs[i].devPtr(),
										 ov->halo()->forces.devPtr() + offsets[i]*ov->objSize,
										 helper->sendBufSizes[i]*sizeof(Force)*ov->objSize,
										 cudaMemcpyDeviceToDevice, stream ) );
	}

	//FIXME hack
	helper->sendBufSizes.uploadToDevice(stream);
	debug2("Will send back forces for %d objects", offsets[27]);
}

void ObjectForcesReverseExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];

	sizes.resize_anew(helper->recvOffsets.size());
	int maximum = 0;
	for (int i=0; i < helper->recvOffsets.size() - 1; i++)
	{
		helper->recvBufs[i].uploadToDevice(stream);
		sizes[i] = (helper->recvOffsets[i+1] - helper->recvOffsets[i]) * ov->objSize;
		maximum = max(sizes[i], maximum);
	}
	sizes.uploadToDevice(stream);

	debug("Updating forces for %d %s objects", helper->recvOffsets[27], ov->name.c_str());

	if (maximum > 0)
	{
		int nthreads = 128;
		dim3 blocks;

		blocks.y = sizes.size();
		blocks.x = getNblocks(maximum, nthreads);

		addHaloForces<<< blocks, nthreads, 0, stream >>> (
				(const float4**)helper->recvAddrs.devPtr(),                        /* source */
				(const int**)entangledHaloExchanger->getOriginAddrs(id).devPtr(),  /* destination ids here */
				(float4*)ov->local()->forces.devPtr(),                             /* add to */
				sizes.devPtr());
	}
}





