#include "object_forces_reverse_exchanger.h"

#include "object_halo_exchanger.h"

#include <core/utils/kernel_launch.h>
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
__global__ void addHaloForces(const float4* recvForces, const int* origins, float4* forces, int np)
{
	const int srcId = blockIdx.x*blockDim.x + threadIdx.x;
	if (srcId >= np) return;
	
	const int dstId = origins[srcId];
	Float3_int extraFrc( recvForces[srcId] );

	atomicAdd(forces + dstId, extraFrc.v);
}

//===============================================================================================
// Member functions
//===============================================================================================

bool ObjectForcesReverseExchanger::needExchange(int id)
{
	return true;
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

	for (int i=0; i < helper->nBuffers; i++)
		helper->sendSizes[i] = offsets[i+1] - offsets[i];

	helper->makeSendOffsets();
	helper->resizeSendBuf();


	CUDA_Check( cudaMemcpyAsync( helper->sendBuf.devPtr(),
								 ov->halo()->forces.devPtr(),
								 helper->sendBuf.size(), cudaMemcpyDeviceToDevice, stream ) );

	debug2("Will send back forces for %d objects", offsets[27]);
}

void ObjectForcesReverseExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];

	int totalRecvd = helper->recvOffsets[helper->nBuffers];
	auto& origins = entangledHaloExchanger->getOrigins(id);

	debug("Updating forces for %d %s objects", totalRecvd, ov->name.c_str());

	const int nthreads = 128;
	SAFE_KERNEL_LAUNCH(
			addHaloForces,
			getNblocks(totalRecvd*ov->objSize, nthreads), nthreads, 0, stream,
			(const float4*)helper->recvBuf.devPtr(),     /* source */
			(const int*)origins.devPtr(),                /* destination ids here */
			(float4*)ov->local()->forces.devPtr(),       /* add to */
			totalRecvd*ov->objSize );
}





