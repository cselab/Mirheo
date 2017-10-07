#include "object_redistributor.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/logger.h>
#include <core/cuda_common.h>

template<bool QUERY>
__global__ void getExitingObjects(const OVviewWithExtraData ovView, const ROVview rovView, char** dests, int* sendBufSizes /*, int* haloParticleIds*/)
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
		shDstObjId = atomicAdd(sendBufSizes + bufId, 1);

	if (QUERY)
		return;

	__syncthreads();

//	if (tid == 0)
//		printf("REDIST  obj  %d  to redist  %d  [%f %f %f] - [%f %f %f]  %d %d %d\n", ovView.ids[objId], bufId,
//				prop.low.x, prop.low.y, prop.low.z, prop.high.x, prop.high.y, prop.high.z, cx, cy, cz);

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
	auto ovView = create_OVviewWithExtraData(ov, ov->local(), stream);
	helper->setDatumSize(ovView.packedObjSize_byte);
	helper->sendBufSizes.clear(stream);

	debug2("Preparing %s halo on the device", ov->name.c_str());

	const int nthreads = 128;
	if (ovView.nObjects > 0)
	{
		// FIXME: this is a hack
		auto rovView = create_ROVview(nullptr, nullptr);
		RigidObjectVector* rov;
		if ( (rov = dynamic_cast<RigidObjectVector*>(ov)) != 0 )
			rovView = create_ROVview(rov, rov->local());

		getExitingObjects<true>  <<< ovView.nObjects, nthreads, 0, stream >>> (
				ovView, rovView, helper->sendAddrs.devPtr(), helper->sendBufSizes.devPtr());

		helper->sendBufSizes.downloadFromDevice(stream);
		helper->resizeSendBufs();

		helper->sendBufSizes.clearDevice(stream);
		getExitingObjects<false> <<< lov->nObjects, nthreads, 0, stream >>> (
				ovView, rovView, helper->sendAddrs.devPtr(), helper->sendBufSizes.devPtr());

		// Unpack the central buffer into the object vector itself
		int nObjs = helper->sendBufSizes[13];
		if (nObjs > 0)
			unpackObject<<< nObjs, nthreads, 0, stream >>> ( (float4*)helper->sendBufs[13].devPtr(), 0, ovView );

		lov->resize(helper->sendBufSizes[13]*ov->objSize, stream);
	}
}

void ObjectRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
	auto ov = objects[id];
	auto helper = helpers[id];

	int oldNObjs = ov->local()->nObjects;
	int objSize = ov->objSize;

	ov->local()->resize_anew(ov->local()->size() + helper->recvOffsets[27] * objSize);
	auto ovView = create_OVviewWithExtraData(ov, ov->local(), stream);

	// TODO: combine in one unpack call
	const int nthreads = 128;
	for (int i=0; i < 27; i++)
	{
		const int nObjs = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (nObjs > 0)
		{
			helper->recvBufs[i].uploadToDevice(stream);
			unpackObject<<< nObjs, nthreads, 0, stream >>> (
					(float4*)helper->recvBufs[i].devPtr(), oldNObjs+helper->recvOffsets[i], ovView);
		}
	}
}



