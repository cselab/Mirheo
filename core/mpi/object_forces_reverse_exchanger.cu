#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/cuda_common.h>

#include <core/mpi/object_forces_reverse_exchanger.h>

#include <vector>
#include <algorithm>
#include <limits>


__global__ void addHaloForces(const float4* haloForces, const float4* halo, float4* forces, int n)
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


void ObjectForcesReverseExchanger::attach(ObjectVector* ov, int* offsetPtr)
{
	objects.push_back(ov);
	offsetPtrs.push_back(offsetPtr)

	const float objPerCell = 0.1f;
	const int maxdim = std::max({ov->localDomainSize.x, ov->localDomainSize.y, ov->localDomainSize.z});

	const int sizes[3] = { (int)(4*objPerCell * maxdim*maxdim + 10),
						   (int)(4*objPerCell * maxdim + 10),
						   (int)(4*objPerCell + 10) };


	ExchangeHelper* helper = new ExchangeHelper(ov->name, ov->local()->objSize*sizeof(Force), sizes);
	ov->halo()->pushStream(helper->stream);
	helpers.push_back(helper);
}


void ObjectForcesReverseExchanger::prepareData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];
	auto offsets = offsetPtrs[id];

	debug2("Preparing %s forces to sending back", ov->name.c_str());

	helper->bufSizes.clearDevice(stream);

	for (int i=0; i<27; i++)
	{
		helper->bufSizes[i] = offsets[i+1] - offsets[i];
		if (helper->bufSizes[i] > 0)
			CUDA_Check( cudaMemcpyAsync(ov->halo()->forces.devPtr() + offsets[i]*ov->halo()->objSize,
										helper->sendBufs[i].hostPtr(),
										helper->bufSizes[i]*sizeof(Force)*ov->halo()->objSize,
										cudaMemcpyHostToDevice, stream) );
	}

	helper->bufSizes.uploadToDevice(stream, false);
}

void ObjectForcesReverseExchanger::combineAndUploadData(int id)
{
	auto ov = objects[id];
	auto helper = helpers[id];

	for (int i=0; i < helper->recvOffsets.size(); i++)
	{
		const int msize = helper->recvOffsets[i+1] - helper->recvOffsets[i];

		if (msize > 0)
			CUDA_Check( cudaMemcpyAsync(ov->halo()->forces.devPtr() + helper->recvOffsets[i]*ov->halo()->objSize,
										helper->recvBufs[compactedDirs[i]].hostPtr(),
										msize*sizeof(Force)*ov->halo()->objSize,
										cudaMemcpyHostToDevice, helper->stream) );
	}

	const int np = helper->recvOffsets[27];
	addHaloForces<<< (np+127)/128, 128, 0, helper->stream >>> (
			(float4*)ov->halo()->forces.devPtr(), (float4*)ov->halo()->coosvels->devPtr(), (float4*)ov->local()->forces.devPtr(), np );
}





