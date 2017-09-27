#include "object_redistributor.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/logger.h>
#include <core/cuda_common.h>

template<bool QUERY>
__global__ void getExitingObjects(const OVview ovView, const int64_t dests[27], int sendBufSizes[27] /*, int* haloParticleIds*/)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	const int sh  = tid % 2;

	if (objId >= ovView.nObjects) return;

	// Find to which buffer this object should go
	auto prop = ovView.comAndExtents[objId];
	int cx = 1, cy = 1, cz = 1;

	if (prop.com.x  < -0.5f*ovView.localDomainSize.x) cx = 0;
	if (prop.com.y  < -0.5f*ovView.localDomainSize.y) cy = 0;
	if (prop.com.z  < -0.5f*ovView.localDomainSize.z) cz = 0;

	if (prop.com.x >=  0.5f*ovView.localDomainSize.x) cx = 2;
	if (prop.com.y >=  0.5f*ovView.localDomainSize.y) cy = 2;
	if (prop.com.z >=  0.5f*ovView.localDomainSize.z) cz = 2;

//	if (tid == 0) printf("Obj %d : [%f %f %f] -- [%f %f %f]\n", objId,
//			prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z);


	const int bufId = (cz*3 + cy)*3 + cx;

	__shared__ int shDstObjId;

	const float3 shift{ ovView.localDomainSize.x*(cx-1),
		                ovView.localDomainSize.y*(cy-1),
		                ovView.localDomainSize.z*(cz-1) };

	__syncthreads();
	if (tid == 0)
		shDstObjId = atomicAdd(sendBufSizes + bufId, 1);

	if (QUERY)
		return;

	__syncthreads();

	if (tid == 0)
		printf("obj  %d  to redist  %d  [%f %f %f] - [%f %f %f]  %d %d %d\n", objId, bufId,
				prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z, cx, cy, cz);

	float4* dstAddr = (float4*) (dests[bufId]) + ovView.packedObjSize_byte/sizeof(float4) * shDstObjId;

	for (int pid = tid/2; pid < ovView.objSize; pid += blockDim.x/2)
	{
		const int srcId = objId * ovView.objSize + pid;
		Float3_int data(ovView.particles[2*srcId + sh]);

		if (sh == 0)
			data.v -= shift;

		dstAddr[2*pid + sh] = data.toFloat4();
	}

	// Add extra data at the end of the object
	dstAddr += ovView.objSize*2;
	ovView.packExtraData(objId, (char*)dstAddr);
}


__global__ static void unpackObject(const float4* from, const int startDstObjId, OVview ovView)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	const int sh  = tid % 2;

	for (int pid = tid/2; pid < ovView.objSize; pid += blockDim.x/2)
	{
		const int srcId = objId * ovView.packedObjSize_byte/sizeof(float4) + pid*2;
		float4 data = from[srcId + sh];

		ovView.particles[2*( (startDstObjId+objId)*ovView.objSize + pid ) + sh] = data;
	}

	ovView.unpackExtraData( startDstObjId+objId,
			(char*)from + objId * ovView.packedObjSize_byte + ovView.objSize*sizeof(Particle) );
}


void ObjectRedistributor::attach(ObjectVector* ov, float rc)
{
	objects.push_back(ov);
	ExchangeHelper* helper = new ExchangeHelper(ov->name, ov->local()->packedObjSize_bytes);
	helpers.push_back(helper);
}


void ObjectRedistributor::prepareData(int id, cudaStream_t stream)
{
	auto ov  = objects[id];
	auto lov = ov->local();
	auto helper = helpers[id];
	auto ovView = OVview(ov, ov->local());

	debug2("Preparing %s halo on the device", ov->name.c_str());

	const int nthreads = 128;
	if (ov->local()->nObjects > 0)
	{

		helper->sendBufSizes.clearDevice(stream);
		getExitingObjects<true>  <<< ovView.nObjects, nthreads, 0, stream >>> (
				ovView, (int64_t*)helper->sendAddrs.devPtr(), helper->sendBufSizes.devPtr());

		helper->sendBufSizes.downloadFromDevice(stream);
		helper->resizeSendBufs(stream);

		helper->sendBufSizes.clearDevice(stream);
		getExitingObjects<false> <<< lov->nObjects, nthreads, 0, stream >>> (
				ovView, (int64_t*)helper->sendAddrs.devPtr(), helper->sendBufSizes.devPtr());

		// Unpack the central buffer into the object vector itself
		int nObjs = helper->sendBufSizes[13];
		unpackObject<<< nObjs, nthreads, 0, stream >>> (
				(float4*)helper->sendBufs[13].devPtr(), 0, ovView);

		lov->resize(helper->sendBufSizes[13]*ov->objSize, stream);
	}
}

void ObjectRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];
	auto ovView = OVview(ov, ov->local());

	int oldNObjs = ov->local()->nObjects;
	int objSize = ov->objSize;

	ov->local()->resize(ov->local()->size() + helper->recvOffsets[27] * objSize, stream, ResizeKind::resizeAnew);

	// TODO: combine in one unpack call
	const int nthreads = 128;
	for (int i=0; i < 27; i++)
	{
		const int nObjs = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (nObjs > 0)
		{
			int        nPtrs = ov->local()->extraDataPtrs.size();
			int totSize_byte = ov->local()->packedObjSize_bytes;

			helper->recvBufs[i].uploadToDevice(stream);
			unpackObject<<< nObjs, nthreads, 0, stream >>> (
					(float4*)helper->recvBufs[i].devPtr(), oldNObjs+helper->recvOffsets[i], ovView);
		}
	}
}



