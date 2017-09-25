#include "object_redistributor.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/logger.h>
#include <core/cuda_common.h>

__device__ static inline void packExtraData(int objId, char** extraData, int nPtrsPerObj, const int* dataSizes, char* destanation)
{
	int baseId = 0;

	for (int ptrId = 0; ptrId < nPtrsPerObj; ptrId++)
		{
			const int size = dataSizes[ptrId];
			for (int i = threadIdx.x; i < size; i += blockDim.x)
				destanation[baseId+i] = extraData[ptrId][objId*size + i];

			baseId += dataSizes[ptrId];
		}
}

__device__ static inline void unpackExtraData(int objId, char** extraData, int nPtrsPerObj, const int* dataSizes, const char* source)
{
	int baseId = 0;

	for (int ptrId = 0; ptrId < nPtrsPerObj; ptrId++)
	{
		const int size = dataSizes[ptrId];
		for (int i = threadIdx.x; i < size; i += blockDim.x)
			extraData[ptrId][objId*size + i] = source[baseId+i];

		baseId += dataSizes[ptrId];
	}
}


template<bool QUERY>
__global__ void getExitingObjects(const float4* __restrict__ coosvels, const LocalObjectVector::COMandExtent* props, const int nObj, const int objSize,
		const float3 localDomainSize,
		const int64_t dests[27], int sendBufSizes[27], /*int* haloParticleIds,*/
		const int packedObjSize_byte, char** extraData, int nPtrsPerObj, const int* dataSizes)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	const int sh  = tid % 2;

	if (objId >= nObj) return;

	// Find to which buffer this object should go
	auto prop = props[objId];
	int cx = 1, cy = 1, cz = 1;

	if (prop.com.x  < -0.5*localDomainSize.x) cx = 0;
	if (prop.com.y  < -0.5*localDomainSize.y) cy = 0;
	if (prop.com.z  < -0.5*localDomainSize.z) cz = 0;

	if (prop.com.x >=  0.5*localDomainSize.x) cx = 2;
	if (prop.com.y >=  0.5*localDomainSize.y) cy = 2;
	if (prop.com.z >=  0.5*localDomainSize.z) cz = 2;

//	if (tid == 0) printf("Obj %d : [%f %f %f] -- [%f %f %f]\n", objId,
//			prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z);


	const int bufId = (cz*3 + cy)*3 + cx;

	__shared__ int shDstObjId;

	const float3 shift{ localDomainSize.x*(cx-1),
						localDomainSize.y*(cy-1),
						localDomainSize.z*(cz-1) };

	__syncthreads();
	if (tid == 0)
		shDstObjId = atomicAdd(sendBufSizes + bufId, 1);

	if (QUERY)
		return;

	__syncthreads();

//		if (tid == 0)
//			//if (objId == 5)
//				printf("obj  %d  to redist  %d  [%f %f %f] - [%f %f %f]  %d %d %d\n", objId, bufId,
//						prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z, cx, cy, cz);

	float4* dstAddr = (float4*) (dests[bufId]) + packedObjSize_byte/sizeof(float4) * shDstObjId;

	for (int pid = tid/2; pid < objSize; pid += blockDim.x/2)
	{
		const int srcId = objId * objSize + pid;
		Float3_int data(coosvels[2*srcId + sh]);

		if (sh == 0) data.v -= shift;

		dstAddr[2*pid + sh] = data.toFloat4();
	}

	// Add extra data at the end of the object
	dstAddr += objSize*2;
	packExtraData(objId, extraData, nPtrsPerObj, dataSizes, (char*)dstAddr);
}


__global__ static void unpackObject(const float4* from, float4* to, const int objSize, const int packedObjSize_byte, const int nObj,
		char** extraData, int nPtrsPerObj, const int* dataSizes)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	const int sh  = tid % 2;

	for (int pid = tid/2; pid < objSize; pid += blockDim.x/2)
	{
		const int srcId = objId * packedObjSize_byte/sizeof(float4) + pid*2;
		float4 data = from[srcId + sh];

		to[2*(objId*objSize + pid) + sh] = data;
	}

	unpackExtraData( objId, extraData, nPtrsPerObj, dataSizes, (char*)from + objId * packedObjSize_byte + objSize*sizeof(Particle) );
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

	debug2("Preparing %s halo on the device", ov->name.c_str());

	const int nthreads = 128;
	if (ov->local()->nObjects > 0)
	{
		int nPtrs  = lov->extraDataPtrs.size();
		int totSize_byte = lov->packedObjSize_bytes;

		helper->sendBufSizes.clearDevice(stream);
		getExitingObjects<true>  <<< lov->nObjects, nthreads, 0, stream >>> (
				(float4*)lov->coosvels.devPtr(), lov->comAndExtents.devPtr(),
				lov->nObjects, lov->objSize, ov->localDomainSize,
				(int64_t*)helper->sendAddrs.devPtr(), helper->sendBufSizes.devPtr(),
				totSize_byte, lov->extraDataPtrs.devPtr(), nPtrs, lov->extraDataSizes.devPtr());

		helper->sendBufSizes.downloadFromDevice(stream);
		helper->resizeSendBufs(stream);

		helper->sendBufSizes.clearDevice(stream);
		getExitingObjects<false> <<< lov->nObjects, nthreads, 0, stream >>> (
				(float4*)lov->coosvels.devPtr(), lov->comAndExtents.devPtr(),
				lov->nObjects, lov->objSize, ov->localDomainSize,
				(int64_t*)helper->sendAddrs.devPtr(), helper->sendBufSizes.devPtr(),
				totSize_byte, lov->extraDataPtrs.devPtr(), nPtrs, lov->extraDataSizes.devPtr());

		// Unpack the central buffer into the object vector itself
		int nObjs = helper->sendBufSizes[13];
		unpackObject<<< nObjs, nthreads, 0, stream >>> (
				(float4*)helper->sendBufs[13].devPtr(), (float4*)lov->coosvels.devPtr(), lov->objSize, lov->packedObjSize_bytes,
				helper->sendBufSizes[13],
				lov->extraDataPtrs.devPtr(), nPtrs, lov->extraDataSizes.devPtr());

		lov->resize(helper->sendBufSizes[13]*ov->objSize, stream);
	}
}

void ObjectRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];

	ov->halo()->resize(helper->recvOffsets[27] * ov->halo()->objSize, stream, ResizeKind::resizeAnew);
	ov->halo()->resize(helper->recvOffsets[27] * ov->halo()->objSize, stream, ResizeKind::resizeAnew);

	// TODO: combine in one unpack call
	const int nthreads = 128;
	for (int i=0; i < 27; i++)
	{
		const int nObjs = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (nObjs > 0)
		{
			int        nPtrs = ov->local()->extraDataPtrs.size();
			int totSize_byte = ov->local()->packedObjSize_bytes;

			unpackObject<<< nObjs, nthreads, 0, stream >>>
					((float4*)helper->recvBufs[i].devPtr(), (float4*)(ov->halo()->coosvels.devPtr() + helper->recvOffsets[i]*nObjs), ov->local()->objSize, totSize_byte, nObjs,
					 ov->halo()->extraDataPtrs.devPtr(), nPtrs, ov->halo()->extraDataSizes.devPtr());
		}
	}
}



