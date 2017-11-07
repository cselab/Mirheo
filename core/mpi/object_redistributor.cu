#include "object_redistributor.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

template<bool QUERY>
__global__ void getExitingObjects(const DomainInfo domain, const OVviewWithExtraData ovView, BufferOffsetsSizesWrap dataWrap)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	const int sh  = tid % 2;

	if (objId >= ovView.nObjects) return;

	// Find to which buffer this object should go
	auto prop = ovView.comAndExtents[objId];
	int cx = 1, cy = 1, cz = 1;

	if (prop.com.x  < -0.5f*domain.localSize.x) cx = 0;
	if (prop.com.y  < -0.5f*domain.localSize.y) cy = 0;
	if (prop.com.z  < -0.5f*domain.localSize.z) cz = 0;

	if (prop.com.x >=  0.5f*domain.localSize.x) cx = 2;
	if (prop.com.y >=  0.5f*domain.localSize.y) cy = 2;
	if (prop.com.z >=  0.5f*domain.localSize.z) cz = 2;

//	if (tid == 0) printf("Obj %d : [%f %f %f] -- [%f %f %f]\n", ovView.ids[objId],
//			prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z);

	const int bufId = (cz*3 + cy)*3 + cx;

	__shared__ int shDstObjId;

	const float3 shift{ domain.localSize.x*(cx-1),
		                domain.localSize.y*(cy-1),
		                domain.localSize.z*(cz-1) };

	__syncthreads();
	if (tid == 0)
		shDstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

	if (QUERY)
		return;

	__syncthreads();

//	if (tid == 0 && bufId != 13)
//		printf("REDIST  obj  %d  to redist  %d  [%f %f %f] - [%f %f %f]  %d %d %d\n", ovView.ids[objId], bufId,
//				prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z, cx, cy, cz);

	float4* dstAddr = (float4*) ( dataWrap.buffer + ovView.packedObjSize_byte * (dataWrap.offsets[bufId] + shDstObjId) );

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

	if (tid == 0) ovView.applyShift2extraData((char*)dstAddr, shift);
}

__global__ static void unpackObject(const char* from, const int startDstObjId, OVviewWithExtraData ovView)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	const int sh  = tid % 2;

	const float4* srcAddr = (float4*) (from + ovView.packedObjSize_byte * objId);

	for (int pid = tid/2; pid < ovView.objSize; pid += blockDim.x/2)
	{
		const int dstId = (startDstObjId+objId)*ovView.objSize + pid;
		ovView.particles[2*dstId + sh] = srcAddr[2*pid + sh];
	}

	ovView.unpackExtraData( startDstObjId+objId, (char*)(srcAddr + 2*ovView.objSize));
}

//===============================================================================================
// Member functions
//===============================================================================================

bool ObjectRedistributor::needExchange(int id)
{
	return !objects[id]->redistValid;
}

void ObjectRedistributor::attach(ObjectVector* ov, float rc)
{
	objects.push_back(ov);
	ExchangeHelper* helper = new ExchangeHelper(ov->name);
	helpers.push_back(helper);
}


// TODO finally split all this shit
void ObjectRedistributor::prepareData(int id, cudaStream_t stream)
{
	auto ov  = objects[id];
	auto lov = ov->local();
	auto helper = helpers[id];
	OVviewWithExtraData ovView(ov, ov->local(), stream);
	helper->setDatumSize(ovView.packedObjSize_byte);

	debug2("Preparing %s halo on the device", ov->name.c_str());
	const int nthreads = 256;

	// Prepare sizes
	helper->sendSizes.clear(stream);
	if (ovView.nObjects > 0)
	{
		SAFE_KERNEL_LAUNCH(
				getExitingObjects<true>,
				ovView.nObjects, nthreads, 0, stream,
				ov->domain, ovView, helper->wrapSendData() );

		helper->makeSendOffsets_Dev2Dev(stream);
	}


	// Early termination - no redistribution
	int nObjs = helper->sendSizes[13];

	if (nObjs == ovView.nObjects)
	{
		helper->sendSizes[13] = 0;
		helper->makeSendOffsets();
		return;
	}


	// Gather data
	helper->resizeSendBuf();
	helper->sendSizes.clearDevice(stream);
	SAFE_KERNEL_LAUNCH(
			getExitingObjects<false>,
			lov->nObjects, nthreads, 0, stream,
			ov->domain, ovView, helper->wrapSendData() );


	// Unpack the central buffer into the object vector itself
	lov->resize_anew(nObjs*ov->objSize);
	ovView = OVviewWithExtraData(ov, ov->local(), stream);

	SAFE_KERNEL_LAUNCH(
			unpackObject,
			nObjs, nthreads, 0, stream,
			helper->sendBuf.devPtr() + helper->sendOffsets[13] * ovView.packedObjSize_byte, 0, ovView );


	// Finally need to compact the buffers
	// TODO: remove this, own buffer should be last
//	if (helper->sendSizes[13] > 0)
//	{
//		int sizeAfter = helper->sendOffsets[helper->nBuffers] - helper->sendOffsets[14];
//
//		// memory may overlap (very rarely), take care of that
//		if (sizeAfter > helper->sendSizes[13])
//		{
//			DeviceBuffer<char> tmp(sizeAfter * helper->datumSize);
//
//			CUDA_Check( cudaMemcpyAsync(
//					tmp.devPtr(),
//					helper->sendBuf.devPtr() + helper->sendOffsets[14],
//					sizeAfter * helper->datumSize,
//					cudaMemcpyDeviceToDevice, stream));
//
//			CUDA_Check( cudaMemcpyAsync(
//					helper->sendBuf.devPtr() + helper->sendOffsets[13],
//					tmp.devPtr(),
//					sizeAfter * helper->datumSize,
//					cudaMemcpyDeviceToDevice, stream));
//		}
//		else // non-overlapping
//		{
//			CUDA_Check( cudaMemcpyAsync(
//					helper->sendBuf.devPtr() + helper->sendOffsets[13],
//					helper->sendBuf.devPtr() + helper->sendOffsets[14],  /* 14 !! */
//					sizeAfter * helper->datumSize,
//					cudaMemcpyDeviceToDevice, stream));
//		}
//
//		helper->sendSizes[13] = 0;
//		helper->makeSendOffsets();
//		helper->resizeSendBuf();   // resize_anew, but strictly smaller size => fine
//	}
}

void ObjectRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];

	int oldNObjs = ov->local()->nObjects;
	int objSize = ov->objSize;

	int totalRecvd = helper->recvOffsets[helper->nBuffers];

	ov->local()->resize(ov->local()->size() + totalRecvd * objSize, stream);
	OVviewWithExtraData ovView(ov, ov->local(), stream);

	const int nthreads = 128;
	SAFE_KERNEL_LAUNCH(
			unpackObject,
			totalRecvd, nthreads, 0, stream,
			helper->recvBuf.devPtr(), oldNObjs, ovView );

	ov->redistValid = true;

	// Particles may have migrated, rebuild cell-lists
	if (totalRecvd > 0)	ov->cellListStamp++;
}



