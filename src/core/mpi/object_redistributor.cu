#include "object_redistributor.h"
#include "exchange_helpers.h"
#include "fragments_mapping.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/pvs/extra_data/packers.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

enum class PackMode
{
    Query, Pack
};

template <PackMode packMode>
__global__ void getExitingObjects(const DomainInfo domain, OVview view, const ObjectPacker packer, BufferOffsetsSizesWrap dataWrap)
{
    const int objId = blockIdx.x;
    const int tid = threadIdx.x;

    if (objId >= view.nObjects) return;

    // Find to which buffer this object should go
    auto prop = view.comAndExtents[objId];
    int dx = 0, dy = 0, dz = 0;

    if (prop.com.x  < -0.5f*domain.localSize.x) dx = -1;
    if (prop.com.y  < -0.5f*domain.localSize.y) dy = -1;
    if (prop.com.z  < -0.5f*domain.localSize.z) dz = -1;

    if (prop.com.x >=  0.5f*domain.localSize.x) dx = 1;
    if (prop.com.y >=  0.5f*domain.localSize.y) dy = 1;
    if (prop.com.z >=  0.5f*domain.localSize.z) dz = 1;

    const int bufId = FragmentMapping::getId(dx, dy, dz);

    __shared__ int shDstObjId;

    const float3 shift{ domain.localSize.x * dx,
                        domain.localSize.y * dy,
                        domain.localSize.z * dz };

    __syncthreads();
    if (tid == 0)
        shDstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

    if (packMode == PackMode::Query) {
        return;
    }
    else {
        __syncthreads();

        char* dstAddr = dataWrap.buffer + packer.totalPackedSize_byte * (dataWrap.offsets[bufId] + shDstObjId);

        for (int pid = tid; pid < view.objSize; pid += blockDim.x)
        {
            const int srcPid = objId * view.objSize + pid;
            packer.part.packShift(srcPid, dstAddr + pid*packer.part.packedSize_byte, -shift);
        }

        dstAddr += view.objSize * packer.part.packedSize_byte;
        if (tid == 0) packer.obj.packShift(objId, dstAddr, -shift);
    }
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

void ObjectRedistributor::attach(ObjectVector* ov)
{
    objects.push_back(ov);

    auto helper = std::make_unique<ExchangeHelper>(ov->name);
    helpers.push_back(std::move(helper));

    info("The Object vector '%s' was attached", ov->name.c_str());
}


void ObjectRedistributor::prepareSizes(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto lov = ov->local();
    auto helper = helpers[id].get();
    auto bulkId = helper->bulkId;
    
    ov->findExtentAndCOM(stream, ParticleVectorType::Local);

    OVview ovView(ov, ov->local());
    ObjectPacker packer(ov, ov->local(), stream);
    helper->setDatumSize(packer.totalPackedSize_byte);

    debug2("Counting exiting objects of '%s'", ov->name.c_str());
    const int nthreads = 256;

    // Prepare sizes
    helper->sendSizes.clear(stream);
    if (ovView.nObjects > 0)
    {
        SAFE_KERNEL_LAUNCH(
                getExitingObjects<PackMode::Query>,
                ovView.nObjects, nthreads, 0, stream,
                ov->state->domain, ovView, packer, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }

    int nObjs = helper->sendSizes[bulkId];
    debug2("%d objects of '%s' will leave", ovView.nObjects - nObjs, ov->name.c_str());

    // Early termination support
    if (nObjs == ovView.nObjects)
    {
        helper->sendSizes[bulkId] = 0;
        helper->computeSendOffsets();
        helper->resizeSendBuf();
    }
}

void ObjectRedistributor::prepareData(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto lov = ov->local();
    auto helper = helpers[id].get();
    auto bulkId = helper->bulkId;

    OVview ovView(ov, ov->local());
    ObjectPacker packer(ov, ov->local(), stream);
    helper->setDatumSize(packer.totalPackedSize_byte);

    const int nthreads = 256;
    int nObjs = helper->sendSizes[bulkId];

    // Early termination - no redistribution
    if (helper->sendOffsets[FragmentMapping::numFragments] == 0)
    {
        debug2("No objects of '%s' leaving, no need to rebuild the object vector", ov->name.c_str());
        return;
    }

    debug2("Downloading %d leaving objects of '%s'", ovView.nObjects - nObjs, ov->name.c_str());

    // Gather data
    helper->resizeSendBuf();
    helper->sendSizes.clearDevice(stream);
    SAFE_KERNEL_LAUNCH(
            getExitingObjects<PackMode::Pack>,
            lov->nObjects, nthreads, 0, stream,
            ov->state->domain, ovView, packer, helper->wrapSendData() );


    // Unpack the central buffer into the object vector itself
    // Renew view and packer, as the ObjectVector may have resized
    lov->resize_anew(nObjs*ov->objSize);
    ovView = OVview(ov, ov->local());
    packer = ObjectPacker(ov, ov->local(), stream);

    SAFE_KERNEL_LAUNCH(
            unpackObject,
            nObjs, nthreads, 0, stream,
            helper->sendBuf.devPtr() + helper->sendOffsets[bulkId] * packer.totalPackedSize_byte, 0, ovView, packer );
                                     
    helper->sendSizes[bulkId] = 0;
    helper->computeSendOffsets();
    helper->resizeSendBuf();
}

void ObjectRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();

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



