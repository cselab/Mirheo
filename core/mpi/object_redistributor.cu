#include "object_redistributor.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/extra_data/packers.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

template<bool QUERY>
__global__ void getExitingObjects(const DomainInfo domain, OVview view, const ObjectPacker packer, BufferOffsetsSizesWrap dataWrap)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;

	if (objId >= view.nObjects) return;

	// Find to which buffer this object should go
	auto prop = view.comAndExtents[objId];
	int cx = 1, cy = 1, cz = 1;

	if (prop.com.x  < -0.5f*domain.localSize.x) cx = 0;
	if (prop.com.y  < -0.5f*domain.localSize.y) cy = 0;
	if (prop.com.z  < -0.5f*domain.localSize.z) cz = 0;

	if (prop.com.x >=  0.5f*domain.localSize.x) cx = 2;
	if (prop.com.y >=  0.5f*domain.localSize.y) cy = 2;
	if (prop.com.z >=  0.5f*domain.localSize.z) cz = 2;

//	if (tid == 0) printf("Obj %d : [%f %f %f] -- [%f %f %f]\n", view.ids[objId],
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
//		printf("REDIST  obj  %d  to redist  %d  [%f %f %f] - [%f %f %f]  %d %d %d\n", view.ids[objId], bufId,
//				prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z, cx, cy, cz);

	char* dstAddr = dataWrap.buffer + packer.totalPackedSize_byte * (dataWrap.offsets[bufId] + shDstObjId);

	for (int pid = tid; pid < view.objSize; pid += blockDim.x)
	{
		const int srcPid = objId * view.objSize + pid;
		packer.part.packShift(srcPid, dstAddr + pid*packer.part.packedSize_byte, -shift);
	}

	dstAddr += view.objSize * packer.part.packedSize_byte;
	if (tid == 0) packer.obj.packShift(objId, dstAddr, -shift);
}

__global__ static void unpackObject(const char* from, const int startDstObjId, OVview view, ObjectPacker packer)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;

	const char* srcAddr = from + packer.totalPackedSize_byte * objId;

	for (int pid = tid; pid < view.objSize; pid += blockDim.x)
	{
		const int dstId = (startDstObjId+objId)*view.objSize + pid;
		packer.part.unpack(srcAddr + pid*packer.part.packedSize_byte, dstId);
	}

	srcAddr += view.objSize * packer.part.packedSize_byte;
	if (tid == 0) packer.obj.unpack(srcAddr, startDstObjId+objId);
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

	ov->findExtentAndCOM(stream, true);

	OVview ovView(ov, ov->local());
	ObjectPacker packer(ov, ov->local(), stream);
	helper->setDatumSize(packer.totalPackedSize_byte);

	debug2("Preparing %s halo on the device", ov->name.c_str());
	const int nthreads = 256;

	// Prepare sizes
	helper->sendSizes.clear(stream);
	if (ovView.nObjects > 0)
	{
		SAFE_KERNEL_LAUNCH(
				getExitingObjects<true>,
				ovView.nObjects, nthreads, 0, stream,
				ov->domain, ovView, packer, helper->wrapSendData() );

		helper->makeSendOffsets_Dev2Dev(stream);
	}


	// Early termination - no redistribution
	int nObjs = helper->sendSizes[13];

	if (nObjs == ovView.nObjects)
	{
		debug2("No objects '%s' leaving, no need to rebuild the object vector", ov->name.c_str());

		helper->sendSizes[13] = 0;
		helper->makeSendOffsets();
		helper->resizeSendBuf();

		return;
	}


	// Gather data
	helper->resizeSendBuf();
	helper->sendSizes.clearDevice(stream);
	SAFE_KERNEL_LAUNCH(
			getExitingObjects<false>,
			lov->nObjects, nthreads, 0, stream,
			ov->domain, ovView, packer, helper->wrapSendData() );


	// Unpack the central buffer into the object vector itself
	// Renew view and packer, as the ObjectVector may have resized
	lov->resize_anew(nObjs*ov->objSize);
	ovView = OVview(ov, ov->local());
	packer = ObjectPacker(ov, ov->local(), stream);

	SAFE_KERNEL_LAUNCH(
			unpackObject,
			nObjs, nthreads, 0, stream,
			helper->sendBuf.devPtr() + helper->sendOffsets[13] * packer.totalPackedSize_byte, 0, ovView, packer );


	// Finally need to compact the buffers
	// TODO: remove this, own buffer should be last (performance penalty only, correctness is there)

	// simple workaround when # of remaining >= # of leaving
//	if (helper->sendSizes[13] >= helper->sendOffsets[27]-helper->sendOffsets[14])
//	{
//		CUDA_Check( cudaMemcpyAsync( helper->sendBuf.hostPtr() + helper->sendOffsets[13]*helper->datumSize,
//									 helper->sendBuf.hostPtr() + helper->sendOffsets[14]*helper->datumSize,
//									 (helper->sendOffsets[27]-helper->sendOffsets[14]) * helper->datumSize,
//									 cudaMemcpyDeviceToDevice, stream ) );
//
//		helper->sendSizes[13] = 0;
//		helper->makeSendOffsets();
//		helper->resizeSendBuf();
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
	OVview ovView(ov, ov->local());
	ObjectPacker packer(ov, ov->local(), stream);

	const int nthreads = 64;
	SAFE_KERNEL_LAUNCH(
			unpackObject,
			totalRecvd, nthreads, 0, stream,
			helper->recvBuf.devPtr(), oldNObjs, ovView, packer );

	ov->redistValid = true;

	// Particles may have migrated, rebuild cell-lists
	if (totalRecvd > 0)
	{
		ov->cellListStamp++;
		ov->local()->comExtentValid = false;
	}
}



