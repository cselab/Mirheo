#include "object_halo_extra_exchanger.h"
#include "object_halo_exchanger.h"
#include "exchange_entity.h"
#include "utils/common.h"
#include "utils/fragments_mapping.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/packers/objects.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <algorithm>

namespace mirheo
{

namespace object_halo_extra_exchanger_kernels
{
template <class PackerHandler>
__global__ void pack(DomainInfo domain, PackerHandler packer, const MapEntry *map,
                     BufferOffsetsSizesWrap dataWrap)
{
    const int objId       = blockIdx.x;
    const int numElements = gridDim.x;

    auto mapEntry = map[objId];

    const int bufId    = mapEntry.getBufId();
    const int srcObjId = mapEntry.getId();
    const int dstObjId = objId - dataWrap.offsets[bufId];
    
    auto buffer = dataWrap.getBuffer(bufId);
    auto dir   = fragment_mapping::getDir(bufId);
    auto shift = exchangers_common::getShift(domain.localSize, dir);

    packer.blockPackShift(numElements, buffer, srcObjId, dstObjId, shift);
}

template <class PackerHandler>
__global__ void unpack(BufferOffsetsSizesWrap dataWrap, PackerHandler packer)
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
} // namespace object_halo_extra_exchanger_kernels


ObjectExtraExchanger::ObjectExtraExchanger(ObjectHaloExchanger *entangledHaloExchanger) :
    entangledHaloExchanger_(entangledHaloExchanger)
{}

ObjectExtraExchanger::~ObjectExtraExchanger() = default;

bool ObjectExtraExchanger::needExchange(__UNUSED size_t id)
{
    return true;
}

void ObjectExtraExchanger::attach(ObjectVector *ov, const std::vector<std::string>& extraChannelNames)
{
    size_t id = objects_.size();
    objects_.push_back(ov);

    auto rv = dynamic_cast<RodVector*>(ov);
    
    PackPredicate predicate = [extraChannelNames](const DataManager::NamedChannelDesc& namedDesc)
    {
        return std::find(extraChannelNames.begin(),
                         extraChannelNames.end(),
                         namedDesc.first)
            != extraChannelNames.end();
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

    auto helper = std::make_unique<ExchangeEntity>(ov->getName(), id, packer.get());
    
    packers_  .push_back(std::move(  packer));
    unpackers_.push_back(std::move(unpacker));

    this->addExchangeEntity(std::move(helper));
}

void ObjectExtraExchanger::prepareSizes(size_t id, cudaStream_t stream)
{
    auto helper = getExchangeEntity(id);
    auto packer = packers_[id].get();
    auto ov = objects_[id];

    packer->update(ov->local(), stream);

    const auto& offsets = entangledHaloExchanger_->getSendOffsets(id);

    for (int i = 0; i < helper->nBuffers; ++i)
        helper->send.sizes[i] = offsets[i+1] - offsets[i];
}

void ObjectExtraExchanger::prepareData(size_t id, cudaStream_t stream)
{
    auto ov     = objects_[id];
    auto helper = getExchangeEntity(id);
    auto packer = packers_[id].get();
    const auto& map = entangledHaloExchanger_->getMap(id);

    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf();

    const int nthreads = 256;
    const int nblocks = static_cast<int>(map.size());
    
    mpark::visit([&](auto packerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            object_halo_extra_exchanger_kernels::pack,
            nblocks, nthreads, 0, stream,
            ov->getState()->domain, packerHandler, map.devPtr(),
            helper->wrapSendData() );
    }, exchangers_common::getHandler(packer));
}

void ObjectExtraExchanger::combineAndUploadData(size_t id, cudaStream_t stream)
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
    const int nblocks  = totalRecvd;
    const size_t shMemSize = offsets.size() * sizeof(offsets[0]);

    mpark::visit([&](auto unpackerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            object_halo_extra_exchanger_kernels::unpack,
            nblocks, nthreads, shMemSize, stream,
            helper->wrapRecvData(), unpackerHandler );
    }, exchangers_common::getHandler(unpacker));
}

} // namespace mirheo
