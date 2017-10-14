#include "object_redistributor.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

template<bool QUERY>
__global__ void getExitingObjects(const OVviewWithExtraData ovView, const ROVview rovView, BufferOffsetsSizesWrap dataWrap)
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

//	if (tid == 0) printf("Obj %d : [%f %f %f] -- [%f %f %f]\n", ovView.ids[objId],
//			prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z);

	const int bufId = (cz*3 + cy)*3 + cx;

	__shared__ int shDstObjId;

	const float3 shift{ ovView.localDomainSize.x*(cx-1),
		                ovView.localDomainSize.y*(cy-1),
		                ovView.localDomainSize.z*(cz-1) };

	__syncthreads();
	if (tid == 0)
		shDstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

	if (QUERY)
		return;

	__syncthreads();

//	if (tid == 0)
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

	if (rovView.objSize == ovView.objSize && tid == 0)
		rovView.applyShift2extraData((char*)dstAddr, shift);
}

__global__ static void unpackObject(const float4* from, const int startDstObjId, OVviewWithExtraData ovView)
{
	const int objId = blockIdx.x;
	const int tid = threadIdx.x;
	const int sh  = tid % 2;

	const float4* srcAddr = from + ovView.packedObjSize_byte/sizeof(float4) * objId;

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

void ObjectRedistributor::prepareData(int id, cudaStream_t stream)
{
	auto ov  = objects[id];
	auto lov = ov->local();
	auto helper = helpers[id];
	OVviewWithExtraData ovView(ov, ov->local(), stream);
	helper->setDatumSize(ovView.packedObjSize_byte);

	debug2("Preparing %s halo on the device", ov->name.c_str());

	helper->sendSizes.clear(stream);
	if (ovView.nObjects > 0)
	{
		const int nthreads = 256;

		// FIXME: this is a hack
		ROVview rovView(nullptr, nullptr);
		RigidObjectVector* rov;
		if ( (rov = dynamic_cast<RigidObjectVector*>(ov)) != 0 )
			rovView = ROVview(rov, rov->local());

		SAFE_KERNEL_LAUNCH(
				getExitingObjects<true>,
				ovView.nObjects, nthreads, 0, stream,
				ovView, rovView, helper->wrapSendData() );

		helper->makeSendOffsets_Dev2Dev(stream);
		helper->resizeSendBuf();

		helper->sendSizes.clearDevice(stream);
		SAFE_KERNEL_LAUNCH(
				getExitingObjects<false>,
				lov->nObjects, nthreads, 0, stream,
				ovView, rovView, helper->wrapSendData() );
	}

	// Unpack the central buffer into the object vector itself
	int nObjs = helper->sendSizes[13];
	lov->resize_anew(nObjs*ov->objSize);
	ovView = OVviewWithExtraData(ov, ov->local(), stream);

	const int nthreads = 128;
	SAFE_KERNEL_LAUNCH(
			unpackObject,
			nObjs, nthreads, 0, stream,
			(float4*) (helper->sendBuf.devPtr() + helper->sendOffsets[13] * ovView.packedObjSize_byte), 0, ovView );
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
			(float4*)helper->recvBuf.devPtr(), oldNObjs, ovView );

	ov->redistValid = true;
}



