#include "object_reverse_exchanger.h"
#include "object_halo_exchanger.h"
#include "exchange_helpers.h"
#include "utils/common.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/packers/objects.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <algorithm>

namespace mirheo
{

namespace ObjectReverseExchangerKernels
{

template <class PackerHandler>
__global__ void reversePack(BufferOffsetsSizesWrap dataWrap, PackerHandler packer)
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

    const int dstObjId = objId - offsets[bufId];
    const int srcObjId = objId;

    packer.blockPack(numElements, buffer, srcObjId, dstObjId);
}

template <class PackerHandler>
__global__ void reverseUnpackAndAdd(PackerHandler packer, const MapEntry *map,
                                    BufferOffsetsSizesWrap dataWrap)
{
    constexpr real eps = 1e-6_r;
    const int objId       = blockIdx.x;
    
    const MapEntry mapEntry = map[objId];
    const int bufId    = mapEntry.getBufId();
    const int dstObjId = mapEntry.getId();
    const int srcObjId = objId - dataWrap.offsets[bufId];
    const int numElements = dataWrap.sizes[bufId];
    
    auto buffer = dataWrap.getBuffer(bufId);

    packer.blockUnpackAddNonZero(numElements, buffer, srcObjId, dstObjId, eps);
}

} // namespace ObjectReverseExchangerKernels


ObjectReverseExchanger::ObjectReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger) :
    entangledHaloExchanger_(entangledHaloExchanger)
{}

ObjectReverseExchanger::~ObjectReverseExchanger() = default;

void ObjectReverseExchanger::attach(ObjectVector *ov, std::vector<std::string> channelNames)
{
    const size_t id = objects_.size();
    objects_.push_back(ov);

    auto rv = dynamic_cast<RodVector*>(ov);
    
    PackPredicate predicate = [channelNames](const DataManager::NamedChannelDesc& namedDesc)
    {
        return std::find(channelNames.begin(),
                         channelNames.end(),
                         namedDesc.first)
            != channelNames.end();
    };

    std::unique_ptr<ObjectPacker> packer, unpacker;

    if (rv == nullptr)
    {
        packer   = std::make_unique<ObjectPacker>(predicate);
        unpacker = std::make_unique<ObjectPacker>(predicate);
    }
    else
    {
        packer   = std::make_unique<RodPacker>(predicate);
        unpacker = std::make_unique<RodPacker>(predicate);
    }
    
    auto helper = std::make_unique<ExchangeHelper>(ov->getName(), id, packer.get());
    
    packers_  .push_back(std::move(  packer));
    unpackers_.push_back(std::move(unpacker));
    this->addExchangeEntity(std::move(  helper));

    std::string allChannelNames = channelNames.size() ? "channels " : "no channels.";
    for (const auto& name : channelNames)
        allChannelNames += "'" + name + "' ";

    info("Object vector '%s' was attached to reverse halo exchanger with %s",
         ov->getCName(), allChannelNames.c_str());
}

bool ObjectReverseExchanger::needExchange(__UNUSED size_t id)
{
    return true;
}

void ObjectReverseExchanger::prepareSizes(size_t id, __UNUSED cudaStream_t stream)
{
    auto  helper  = getExchangeEntity(id);
    auto& offsets = entangledHaloExchanger_->getRecvOffsets(id);
    
    for (int i = 0; i < helper->nBuffers; ++i)
        helper->send.sizes[i] = offsets[i+1] - offsets[i];
}

void ObjectReverseExchanger::prepareData(size_t id, cudaStream_t stream)
{
    auto ov     = objects_[id];
    auto hov    = ov->halo();
    auto helper = getExchangeEntity(id);
    auto packer = packers_[id].get();
    
    debug2("Preparing '%s' data to reverse send", ov->getCName());

    packer->update(hov, stream);

    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf();

    const auto& offsets = helper->send.offsets;
    const int nSendObj = offsets[helper->nBuffers];
    
    const int nthreads = 256;
    const int nblocks = nSendObj;

    const size_t shMemSize = offsets.size() * sizeof(offsets[0]);

    mpark::visit([&](auto packerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            ObjectReverseExchangerKernels::reversePack,
            nblocks, nthreads, shMemSize, stream,
            helper->wrapSendData(), packerHandler );
    }, ExchangersCommon::getHandler(packer));
    
    debug2("Will send back data for %d objects", nSendObj);
}

void ObjectReverseExchanger::combineAndUploadData(size_t id, cudaStream_t stream)
{
    auto ov       = objects_[id];
    auto lov      = ov->local();
    auto helper   = getExchangeEntity(id);
    auto unpacker = unpackers_[id].get();

    unpacker->update(lov, stream);

    const int totalRecvd = helper->recv.offsets[helper->nBuffers];
    auto& map = entangledHaloExchanger_->getMap(id);

    debug("Updating data for %d '%s' objects", totalRecvd, ov->getCName());

    const int nthreads = 256;
        
    mpark::visit([&](auto unpackerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            ObjectReverseExchangerKernels::reverseUnpackAndAdd,
            static_cast<int>(map.size()), nthreads, 0, stream,
            unpackerHandler, map.devPtr(),
            helper->wrapRecvData());
    }, ExchangersCommon::getHandler(unpacker));
}

} // namespace mirheo
