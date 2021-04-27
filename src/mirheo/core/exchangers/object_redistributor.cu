// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "object_redistributor.h"

#include "exchange_entity.h"
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

namespace objec_redistributor_kernels
{

template <PackMode packMode, class PackerHandler>
__global__ void getExitingObjects(DomainInfo domain, OVview view,
                                  PackerHandler packer, BufferOffsetsSizesWrap dataWrap)
{
    const int objId = blockIdx.x;
    const int tid   = threadIdx.x;

    // Find to which buffer this object should go
    const auto prop = view.comAndExtents[objId];
    const int3 dir  = exchangers_common::getDirection(prop.com, domain.localSize);

    const int bufId = fragment_mapping::getId(dir);

    // Object is marked for deletion if all its particles are marked. The easiest
    // way to check that is to check the COM (which is computed anyway).
    // However, we must allow for some numerical error. Namely, with mark_val
    // of -900.0, checking exactly prop.com.x == mark_val could break already
    // for objects with 74k particles (in single precision).
    const bool isMarked = std::fabs(prop.com.x - Real3_int::mark_val) < 1.0e-2_r;
    if (isMarked)
        return;  // 1 block == 1 object, all threads agree on the branch.

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

        const auto shift = exchangers_common::getShift(domain.localSize, dir);

        auto buffer = dataWrap.getBuffer(bufId);
        const int numElements = dataWrap.offsets[bufId+1] - dataWrap.offsets[bufId];

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

} // namespace objec_redistributor_kernels

ObjectRedistributor::ObjectRedistributor() = default;
ObjectRedistributor::~ObjectRedistributor() = default;

bool ObjectRedistributor::needExchange(size_t id)
{
    return !objects_[id]->redistValid;
}

void ObjectRedistributor::attach(ObjectVector *ov)
{
    const size_t id = objects_.size();
    objects_.push_back(ov);

    PackPredicate predicate = [](const DataManager::NamedChannelDesc& namedDesc)
    {
        return namedDesc.second->persistence == DataManager::PersistenceMode::Active;
    };

    std::unique_ptr<ObjectPacker> packer;
    if (auto rv = dynamic_cast<RodVector*>(ov)) packer = std::make_unique<RodPacker>   (predicate);
    else                                        packer = std::make_unique<ObjectPacker>(predicate);

    auto helper = std::make_unique<ExchangeEntity>(ov->getName(), id, packer.get());

    packers_.push_back(std::move(packer));
    this->addExchangeEntity(std::move(helper));

    info("The Object vector '%s' was attached to redistributor", ov->getCName());
}


void ObjectRedistributor::prepareSizes(size_t id, cudaStream_t stream)
{
    auto ov  = objects_[id];
    auto lov = ov->local();
    auto helper = getExchangeEntity(id);
    auto packer = packers_[id].get();
    auto bulkId = helper->bulkId;

    ov->findExtentAndCOM(stream, ParticleVectorLocality::Local);

    OVview ovView(ov, lov);

    debug2("Counting exiting objects of '%s'", ov->getCName());

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
                objec_redistributor_kernels::getExitingObjects<PackMode::Query>,
                nblocks, nthreads, 0, stream,
                ov->getState()->domain, ovView, packerHandler, helper->wrapSendData() );
        }, exchangers_common::getHandler(packer));

        helper->computeSendOffsets_Dev2Dev(stream);
    }

    const int nObjs = helper->send.sizes[bulkId];
    debug2("%d objects of '%s' will leave or be removed",
           ovView.nObjects - nObjs, ov->getCName());
}

void ObjectRedistributor::prepareData(size_t id, cudaStream_t stream)
{
    auto ov  = objects_[id];
    auto lov = ov->local();
    auto helper = getExchangeEntity(id);
    auto bulkId = helper->bulkId;
    auto packer = packers_[id].get();

    OVview ovView(ov, lov);

    int nObjsBulk = helper->send.sizes[bulkId];

    // Early termination - no redistribution
    if (helper->send.sizes[bulkId] == lov->getNumObjects())
    {
        debug2("No objects of '%s' leaving or being removed, no need to rebuild the object vector",
               ov->getCName());
        helper->send.sizes[bulkId] = 0;
        helper->computeSendOffsets();
        helper->send.uploadInfosToDevice(stream);
        helper->resizeSendBuf();
        return;
    }

    debug2("%d leaving or removed objects of '%s'", ovView.nObjects - nObjsBulk,
           ov->getCName());

    // Gather data
    helper->resizeSendBuf();
    helper->send.sizes.clearDevice(stream);

    const int nthreads = 256;
    const int nblocks  = ovView.nObjects;

    mpark::visit([&](auto packerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            objec_redistributor_kernels::getExitingObjects<PackMode::Pack>,
            nblocks, nthreads, 0, stream,
            ov->getState()->domain, ovView, packerHandler, helper->wrapSendData() );
    }, exchangers_common::getHandler(packer));

    // Unpack the central buffer into the object vector itself
    lov->resize_anew(nObjsBulk * ov->getObjectSize());
    packer->update(lov, stream);

    mpark::visit([&](auto packerHandler)
    {
        SAFE_KERNEL_LAUNCH(
             objec_redistributor_kernels::unpackObjects,
             nObjsBulk, nthreads, 0, stream,
             helper->send.getBufferDevPtr(bulkId), 0, packerHandler);
    }, exchangers_common::getHandler(packer));

    helper->send.sizes[bulkId] = 0;
    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf(); // relying here on the fact that bulkId is the last one
}

void ObjectRedistributor::combineAndUploadData(size_t id, cudaStream_t stream)
{
    auto ov     = objects_[id];
    auto lov    = ov->local();
    auto helper = getExchangeEntity(id);
    auto packer = packers_[id].get();

    const int oldNObjs = lov->getNumObjects();
    const int objSize = ov->getObjectSize();

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
                objec_redistributor_kernels::unpackObjects,
                nObjs, nthreads, 0, stream,
                helper->recv.getBufferDevPtr(bufId),
                oldNObjs + helper->recv.offsets[bufId], packerHandler );
        }, exchangers_common::getHandler(packer));
    }

    ov->redistValid = true;
    ov->cellListStamp++;
}

} // namespace mirheo
