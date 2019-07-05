#include "object_redistributor.h"
#include "exchange_helpers.h"
#include "utils/fragments_mapping.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/pvs/packers/objects.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

enum class PackMode
{
    Query, Pack
};

namespace ObjecRedistributorKernels
{

template <PackMode packMode>
__global__ void getExitingObjects(DomainInfo domain, OVview view,
                                  ObjectPackerHandler packer, BufferOffsetsSizesWrap dataWrap)
{
    const int objId = blockIdx.x;
    const int tid   = threadIdx.x;
    
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

    __syncthreads();
    
    if (tid == 0)
        shDstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

    if (packMode == PackMode::Query)
    {
        return;
    }
    else
    {
        __syncthreads();
        
        const float3 shift{ - domain.localSize.x * dx,
                            - domain.localSize.y * dy,
                            - domain.localSize.z * dz };

        auto buffer = dataWrap.buffer + dataWrap.offsetsBytes[bufId];
        int numElements = dataWrap.offsets[bufId+1] - dataWrap.offsets[bufId];

        size_t offsetBytes = 0;
        
        for (int pid = tid; pid < view.objSize; pid += blockDim.x)
        {
            const int srcPid = objId      * view.objSize + pid;
            const int dstPid = shDstObjId * view.objSize + pid;
            
            offsetBytes = packer.particles.packShift(srcPid, dstPid, buffer,
                                                     numElements * view.objSize, shift);
        }

        buffer += offsetBytes;
        
        if (tid == 0)
            packer.objects.packShift(objId, shDstObjId, buffer, numElements, shift);
    }
}

__global__ void unpackObjects(const char *buffer, int startDstObjId,
                              OVview view, ObjectPackerHandler packer)
{
    const int objId = blockIdx.x;
    const int tid   = threadIdx.x;
    const int numElements = gridDim.x;

    const int srcObjId = objId;
    const int dstObjId = objId + startDstObjId;
    
    size_t offsetBytes = 0;
    
    for (int pid = tid; pid < view.objSize; pid += blockDim.x)
    {
        const int dstPid = dstObjId * view.objSize + pid;
        const int srcPid = srcObjId * view.objSize + pid;
        offsetBytes = packer.particles.unpack(srcPid, dstPid, buffer,
                                              numElements * view.objSize);
    }

    buffer += offsetBytes;
    
    if (tid == 0)
        packer.objects.unpack(srcObjId, dstObjId, buffer, numElements);
}

} // namespace ObjecRedistributorKernels

//===============================================================================================
// Member functions
//===============================================================================================

ObjectRedistributor::ObjectRedistributor() = default;
ObjectRedistributor::~ObjectRedistributor() = default;

bool ObjectRedistributor::needExchange(int id)
{
    return !objects[id]->redistValid;
}

void ObjectRedistributor::attach(ObjectVector *ov)
{
    int id = objects.size();
    objects.push_back(ov);

    PackPredicate predicate = [](const DataManager::NamedChannelDesc& namedDesc)
    {
        return (namedDesc.second->persistence == DataManager::PersistenceMode::Persistent) ||
            (namedDesc.first == ChannelNames::positions);
    };
    
    auto packer = std::make_unique<ObjectPacker>(predicate);
    auto helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());

    packers.push_back(std::move(packer));
    helpers.push_back(std::move(helper));

    info("The Object vector '%s' was attached to redistributor", ov->name.c_str());
}


void ObjectRedistributor::prepareSizes(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto lov = ov->local();
    auto helper = helpers[id].get();
    auto packer = packers[id].get();
    auto bulkId = helper->bulkId;
    
    ov->findExtentAndCOM(stream, ParticleVectorType::Local);
    
    OVview ovView(ov, ov->local());

    debug2("Counting exiting objects of '%s'", ov->name.c_str());

    // Prepare sizes
    helper->send.sizes.clear(stream);
    packer->update(ov->local(), stream);
    
    if (ovView.nObjects > 0)
    {
        const int nthreads = 256;
        const int nblocks  = ovView.nObjects;
        
        SAFE_KERNEL_LAUNCH(
            ObjecRedistributorKernels::getExitingObjects<PackMode::Query>,
            nblocks, nthreads, 0, stream,
            ov->state->domain, ovView, packer->handler(), helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }

    int nObjs = helper->send.sizes[bulkId];
    debug2("%d objects of '%s' will leave", ovView.nObjects - nObjs, ov->name.c_str());

    // Early termination support
    if (nObjs == ovView.nObjects)
    {
        helper->send.sizes[bulkId] = 0;
        helper->computeSendOffsets();
        helper->send.uploadInfosToDevice(stream);
        helper->resizeSendBuf();
    }
}

void ObjectRedistributor::prepareData(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto lov = ov->local();
    auto helper = helpers[id].get();
    auto bulkId = helper->bulkId;
    auto packer = packers[id].get();

    OVview ovView(ov, ov->local());

    int nObjsBulk = helper->send.sizes[bulkId];

    // Early termination - no redistribution
    if (helper->send.offsets[helper->nBuffers] == 0)
    {
        debug2("No objects of '%s' leaving, no need to rebuild the object vector", ov->name.c_str());
        return;
    }

    debug2("Downloading %d leaving objects of '%s'", ovView.nObjects - nObjsBulk, ov->name.c_str());

    // Gather data
    helper->resizeSendBuf();
    helper->send.sizes.clearDevice(stream);
    
    const int nthreads = 256;
    const int nblocks  = ovView.nObjects;

    SAFE_KERNEL_LAUNCH(
        ObjecRedistributorKernels::getExitingObjects<PackMode::Pack>,
        nblocks, nthreads, 0, stream,
        ov->state->domain, ovView, packer->handler(), helper->wrapSendData() );    

    // Unpack the central buffer into the object vector itself
    // Renew view and packer, as the ObjectVector may have resized
    lov->resize_anew(nObjsBulk * ov->objSize);

    SAFE_KERNEL_LAUNCH(
         ObjecRedistributorKernels::unpackObjects,
         nObjsBulk, nthreads, 0, stream,
         helper->send.buffer.devPtr() + helper->send.offsetsBytes[bulkId], 0,
         ovView, packer->handler() );
    
    helper->send.sizes[bulkId] = 0;
    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf(); // relying here on the fact that bulkId is the last one
}

void ObjectRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    int oldNObjs = ov->local()->nObjects;
    int objSize = ov->objSize;

    int totalRecvd = helper->recv.offsets[helper->nBuffers];

    ov->local()->resize((oldNObjs + totalRecvd) * objSize, stream);

    OVview ovView(ov, ov->local());

    // TODO separate streams?
    for (int bufId = 0; bufId < helper->nBuffers; ++bufId)
    {
        int nObjs = helper->recv.sizes[bufId];

        if (bufId == helper->bulkId || nObjs == 0) continue;

        const int nthreads = 256;
        
        SAFE_KERNEL_LAUNCH(
            ObjecRedistributorKernels::unpackObjects,
            nObjs, nthreads, 0, stream,
            helper->recv.buffer.devPtr() + helper->recv.offsetsBytes[bufId],
            oldNObjs + helper->recv.offsets[bufId],
            ovView, packer->handler() );
    }

    ov->redistValid = true;

    // Particles may have migrated, rebuild cell-lists
    if (totalRecvd > 0)
        ov->cellListStamp++;
}



