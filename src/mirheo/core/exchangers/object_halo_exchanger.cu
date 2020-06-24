// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "object_halo_exchanger.h"

#include "exchange_entity.h"
#include "utils/common.h"
#include "utils/fragments_mapping.h"

#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/packers/objects.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/cuda_common.h>

#include <algorithm>

namespace mirheo
{

enum class PackMode
{
    Query, Pack
};

namespace object_halo_exchange_kernels
{

template <PackMode packMode, class PackerHandler>
__global__ void getObjectHaloAndMap(DomainInfo domain, OVview view, MapEntry *map,
                                    real rc, PackerHandler packer,
                                    BufferOffsetsSizesWrap dataWrap)
{
    const int objId = blockIdx.x;
    const int tid   = threadIdx.x;

    int nHalos = 0;
    short validHalos[7];

    if (objId < view.nObjects)
    {
        // Find to which halos this object should go
        auto prop = view.comAndExtents[objId];
        int dx = 0, dy = 0, dz = 0;

        if (prop.low.x  < -0.5_r * domain.localSize.x + rc) dx = -1;
        if (prop.low.y  < -0.5_r * domain.localSize.y + rc) dy = -1;
        if (prop.low.z  < -0.5_r * domain.localSize.z + rc) dz = -1;

        if (prop.high.x >  0.5_r * domain.localSize.x - rc) dx = 1;
        if (prop.high.y >  0.5_r * domain.localSize.y - rc) dy = 1;
        if (prop.high.z >  0.5_r * domain.localSize.z - rc) dz = 1;

        for (int ix = math::min(dx, 0); ix <= math::max(dx, 0); ++ix)
            for (int iy = math::min(dy, 0); iy <= math::max(dy, 0); ++iy)
                for (int iz = math::min(dz, 0); iz <= math::max(dz, 0); ++iz)
                {
                    if (ix == 0 && iy == 0 && iz == 0) continue;
                    const int bufId = fragment_mapping::getId(ix, iy, iz);
                    validHalos[nHalos] = bufId;
                    nHalos++;
                }
    }


    // Copy objects to each halo
    __shared__ int shDstObjId;

    for (int i = 0; i < nHalos; ++i)
    {
        const int bufId = validHalos[i];

        __syncthreads();
        if (tid == 0)
            shDstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

        if (packMode == PackMode::Query)
        {
            continue;
        }
        else
        {
            __syncthreads();

            const int3 dir = fragment_mapping::getDir(bufId);
            const auto shift = exchangers_common::getShift(domain.localSize, dir);

            auto buffer = dataWrap.getBuffer(bufId);
            const int numElements = dataWrap.offsets[bufId+1] - dataWrap.offsets[bufId];

            packer.blockPackShift(numElements, buffer, objId, shDstObjId, shift);

            // save map

            const int myOffset = dataWrap.offsets[bufId] + shDstObjId;
            if (tid == 0)
                map[myOffset] = MapEntry(objId, bufId);
        }
    }
}

template <class PackerHandler>
__global__ void unpackObjects(BufferOffsetsSizesWrap dataWrap, PackerHandler packer)
{
    const int objId = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ int offsets[];

    const int nBuffers = dataWrap.nBuffers;

    for (int i = tid; i < nBuffers + 1; i += blockDim.x)
        offsets[i] = dataWrap.offsets[i];
    __syncthreads();

    const int bufId = dispatchThreadsPerBuffer(nBuffers, offsets, objId);
    auto buffer = dataWrap.getBuffer(bufId);
    const int numElements = dataWrap.sizes[bufId];

    const int srcObjId = objId - offsets[bufId];
    const int dstObjId = objId;

    packer.blockUnpack(numElements, buffer, srcObjId, dstObjId);
}

} // namespace object_halo_exchange_kernels


bool ObjectHaloExchanger::needExchange(size_t id)
{
    return !objects_[id]->haloValid;
}

ObjectHaloExchanger::ObjectHaloExchanger() = default;
ObjectHaloExchanger::~ObjectHaloExchanger() = default;

void ObjectHaloExchanger::attach(ObjectVector *ov, real rc, const std::vector<std::string>& extraChannelNames)
{
    const size_t id = objects_.size();
    objects_.push_back(ov);
    rcs_.push_back(rc);

    auto channels = extraChannelNames;
    channels.push_back(channel_names::positions);
    channels.push_back(channel_names::velocities);

    PackPredicate predicate = [channels](const DataManager::NamedChannelDesc& namedDesc)
    {
        return std::find(channels.begin(), channels.end(), namedDesc.first) != channels.end();
    };

    std::unique_ptr<ObjectPacker> packer, unpacker;

    if (auto rv = dynamic_cast<RodVector*>(ov))
    {
        packer   = std::make_unique<RodPacker>(predicate);
        unpacker = std::make_unique<RodPacker>(predicate);
    }
    else
    {
        packer   = std::make_unique<ObjectPacker>(predicate);
        unpacker = std::make_unique<ObjectPacker>(predicate);
    }

    auto helper = std::make_unique<ExchangeEntity>(ov->getName(), id, packer.get());

    packers_  .push_back(std::move(  packer));
    unpackers_.push_back(std::move(unpacker));
    this->addExchangeEntity(std::move(helper));
    maps_.emplace_back();

    std::string allChannelNames = "";
    for (const auto& name : channels)
        allChannelNames += "'" + name + "' ";

    info("Object vector '%s' (rc %f) was attached to halo exchanger with channels %s",
         ov->getCName(), rc, allChannelNames.c_str());
}

void ObjectHaloExchanger::prepareSizes(size_t id, cudaStream_t stream)
{
    auto ov  = objects_[id];
    auto lov = ov->local();
    auto rc  = rcs_[id];
    auto helper = getExchangeEntity(id);
    auto packer = packers_[id].get();

    ov->findExtentAndCOM(stream, ParticleVectorLocality::Local);

    debug2("Counting halo objects of '%s'", ov->getCName());

    OVview ovView(ov, lov);
    helper->send.sizes.clear(stream);
    packer->update(lov, stream);

    if (ovView.nObjects > 0)
    {
        const int nthreads = 256;

        mpark::visit([&](auto packerHandler)
        {
            SAFE_KERNEL_LAUNCH(
                object_halo_exchange_kernels::getObjectHaloAndMap<PackMode::Query>,
                ovView.nObjects, nthreads, 0, stream,
                ov->getState()->domain, ovView, nullptr, rc,
                packerHandler, helper->wrapSendData() );
        }, exchangers_common::getHandler(packer));
    }

    helper->computeSendOffsets_Dev2Dev(stream);
}

void ObjectHaloExchanger::prepareData(size_t id, cudaStream_t stream)
{
    auto ov  = objects_[id];
    auto lov = ov->local();
    auto rc  = rcs_[id];
    auto helper = getExchangeEntity(id);
    auto packer = packers_[id].get();
    auto& map = maps_[id];

    const int nhalo = helper->send.offsets[helper->nBuffers];
    OVview ovView(ov, lov);
    map.resize_anew(nhalo);

    if (ovView.nObjects > 0)
    {
        const int nthreads = 256;
        debug2("Downloading %d halo objects of '%s'", nhalo, ov->getCName());

        helper->resizeSendBuf();
        helper->send.sizes.clearDevice(stream);

        mpark::visit([&](const auto& packerHandler)
        {
            SAFE_KERNEL_LAUNCH(
                object_halo_exchange_kernels::getObjectHaloAndMap<PackMode::Pack>,
                ovView.nObjects, nthreads, 0, stream,
                ov->getState()->domain, ovView, map.devPtr(), rc,
                packerHandler, helper->wrapSendData());
        }, exchangers_common::getHandler(packer));
    }
}

void ObjectHaloExchanger::combineAndUploadData(size_t id, cudaStream_t stream)
{
    auto ov       = objects_[id];
    auto hov      = ov->halo();
    auto helper   = getExchangeEntity(id);
    auto unpacker = unpackers_[id].get();

    const auto& offsets = helper->recv.offsets;
    const int totalRecvd = offsets[helper->nBuffers];

    hov->resize_anew(totalRecvd * ov->getObjectSize());
    unpacker->update(hov, stream);

    const int nthreads = 256;
    const int nblocks = totalRecvd;
    const size_t shMemSize = offsets.size() * sizeof(offsets[0]);

    mpark::visit([&](const auto& unpackerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            object_halo_exchange_kernels::unpackObjects,
            nblocks, nthreads, shMemSize, stream,
            helper->wrapRecvData(), unpackerHandler );
    }, exchangers_common::getHandler(unpacker));
}

PinnedBuffer<int>& ObjectHaloExchanger::getSendOffsets(size_t id)
{
    return getExchangeEntity(id)->send.offsets;
}

PinnedBuffer<int>& ObjectHaloExchanger::getRecvOffsets(size_t id)
{
    return getExchangeEntity(id)->recv.offsets;
}

DeviceBuffer<MapEntry>& ObjectHaloExchanger::getMap(size_t id)
{
    return maps_[id];
}

} // namespace mirheo
