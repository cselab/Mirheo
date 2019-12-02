#include "object_redistributor.h"

#include "exchange_helpers.h"
#include "utils/common.h"
#include "utils/fragments_mapping.h"

#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/pvs/packers/objects.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{

enum class PackMode
{
    Query, Pack
};

namespace ObjecRedistributorKernels
{

template <PackMode packMode, class PackerHandler>
__global__ void getExitingObjects(DomainInfo domain, OVview view,
                                  PackerHandler packer, BufferOffsetsSizesWrap dataWrap)
{
    const int objId = blockIdx.x;
    const int tid   = threadIdx.x;
    
    // Find to which buffer this object should go
    auto prop = view.comAndExtents[objId];
    auto dir  = ExchangersCommon::getDirection(prop.com, domain.localSize);

    const int bufId = FragmentMapping::getId(dir);

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
        
        auto shift = ExchangersCommon::getShift(domain.localSize, dir);

        auto buffer = dataWrap.getBuffer(bufId);
        int numElements = dataWrap.offsets[bufId+1] - dataWrap.offsets[bufId];

        packer.blockPackShift(numElements, buffer, objId, shDstObjId, shift);
    }
}

template <class PackerHandler>
__global__ void unpackObjects(const char *buffer, int startDstObjId, PackerHandler packer)
{
    const int objId = blockIdx.x;
    const int numElements = gridDim.x;

    const int srcObjId = objId;
    const int dstObjId = objId + startDstObjId;

    packer.blockUnpack(numElements, buffer, srcObjId, dstObjId);
}

} // namespace ObjecRedistributorKernels

ObjectRedistributor::ObjectRedistributor() = default;
ObjectRedistributor::~ObjectRedistributor() = default;

bool ObjectRedistributor::needExchange(size_t id)
{
    return !objects[id]->redistValid;
}

void ObjectRedistributor::attach(ObjectVector *ov)
{
    const size_t id = objects.size();
    objects.push_back(ov);

    PackPredicate predicate = [](const DataManager::NamedChannelDesc& namedDesc)
    {
        return namedDesc.second->persistence == DataManager::PersistenceMode::Active;
    };
    
    std::unique_ptr<ObjectPacker> packer;
    if (auto rv = dynamic_cast<RodVector*>(ov)) packer = std::make_unique<RodPacker>   (predicate);
    else                                        packer = std::make_unique<ObjectPacker>(predicate);
    
    auto helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());
    
    packers.push_back(std::move(packer));
    helpers.push_back(std::move(helper));

    info("The Object vector '%s' was attached to redistributor", ov->name.c_str());
}


void ObjectRedistributor::prepareSizes(size_t id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto lov = ov->local();
    auto helper = helpers[id].get();
    auto packer = packers[id].get();
    auto bulkId = helper->bulkId;
    
    ov->findExtentAndCOM(stream, ParticleVectorLocality::Local);
    
    OVview ovView(ov, lov);

    debug2("Counting exiting objects of '%s'", ov->name.c_str());

    // Prepare sizes
    helper->send.sizes.clear(stream);
    packer->update(lov, stream);
    
    if (ovView.nObjects > 0)
    {
        const int nthreads = 256;
        const int nblocks  = ovView.nObjects;

        mpark::visit([&](auto packerHandler)
        {
            SAFE_KERNEL_LAUNCH(
                ObjecRedistributorKernels::getExitingObjects<PackMode::Query>,
                nblocks, nthreads, 0, stream,
                ov->getState()->domain, ovView, packerHandler, helper->wrapSendData() );
        }, ExchangersCommon::getHandler(packer));

        helper->computeSendOffsets_Dev2Dev(stream);
    }

    const int nObjs = helper->send.sizes[bulkId];
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

void ObjectRedistributor::prepareData(size_t id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto lov = ov->local();
    auto helper = helpers[id].get();
    auto bulkId = helper->bulkId;
    auto packer = packers[id].get();

    OVview ovView(ov, lov);

    int nObjsBulk = helper->send.sizes[bulkId];

    // Early termination - no redistribution
    if (helper->send.offsets[helper->nBuffers] == 0)
    {
        debug2("No objects of '%s' leaving, no need to rebuild the object vector",
               ov->name.c_str());
        return;
    }

    debug2("Downloading %d leaving objects of '%s'", ovView.nObjects - nObjsBulk,
           ov->name.c_str());

    // Gather data
    helper->resizeSendBuf();
    helper->send.sizes.clearDevice(stream);
    
    const int nthreads = 256;
    const int nblocks  = ovView.nObjects;

    mpark::visit([&](auto packerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            ObjecRedistributorKernels::getExitingObjects<PackMode::Pack>,
            nblocks, nthreads, 0, stream,
            ov->getState()->domain, ovView, packerHandler, helper->wrapSendData() );
    }, ExchangersCommon::getHandler(packer));

    // Unpack the central buffer into the object vector itself
    lov->resize_anew(nObjsBulk * ov->objSize);
    packer->update(lov, stream);

    mpark::visit([&](auto packerHandler)
    {
        SAFE_KERNEL_LAUNCH(
             ObjecRedistributorKernels::unpackObjects,
             nObjsBulk, nthreads, 0, stream,
             helper->send.getBufferDevPtr(bulkId), 0, packerHandler);
    }, ExchangersCommon::getHandler(packer));
    
    helper->send.sizes[bulkId] = 0;
    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf(); // relying here on the fact that bulkId is the last one
}

void ObjectRedistributor::combineAndUploadData(size_t id, cudaStream_t stream)
{
    auto ov     = objects[id];
    auto lov    = ov->local();
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    int oldNObjs = lov->nObjects;
    int objSize = ov->objSize;

    int totalRecvd = helper->recv.offsets[helper->nBuffers];

    lov->resize((oldNObjs + totalRecvd) * objSize, stream);
    packer->update(lov, stream);

    for (int bufId = 0; bufId < helper->nBuffers; ++bufId)
    {
        int nObjs = helper->recv.sizes[bufId];

        if (bufId == helper->bulkId || nObjs == 0) continue;

        const int nthreads = 256;

        mpark::visit([&](auto packerHandler)
        {
            SAFE_KERNEL_LAUNCH(
                ObjecRedistributorKernels::unpackObjects,
                nObjs, nthreads, 0, stream,
                helper->recv.getBufferDevPtr(bufId),
                oldNObjs + helper->recv.offsets[bufId], packerHandler );
        }, ExchangersCommon::getHandler(packer));
    }

    ov->redistValid = true;
    ov->cellListStamp++;
}

} // namespace mirheo
