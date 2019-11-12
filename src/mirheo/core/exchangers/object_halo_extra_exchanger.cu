#include "object_halo_extra_exchanger.h"
#include "object_halo_exchanger.h"
#include "exchange_helpers.h"
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

namespace ObjectHaloExtraExchangerKernels
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
    auto dir   = FragmentMapping::getDir(bufId);
    auto shift = ExchangersCommon::getShift(domain.localSize, dir);

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
} // namespace ObjectHaloExtraExchangerKernels


ObjectExtraExchanger::ObjectExtraExchanger(ObjectHaloExchanger *entangledHaloExchanger) :
    entangledHaloExchanger(entangledHaloExchanger)
{}

ObjectExtraExchanger::~ObjectExtraExchanger() = default;

bool ObjectExtraExchanger::needExchange(__UNUSED int id)
{
    return true;
}

void ObjectExtraExchanger::attach(ObjectVector *ov, const std::vector<std::string>& extraChannelNames)
{
    int id = objects.size();
    objects.push_back(ov);

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

    auto   helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());
    
    packers  .push_back(std::move(  packer));
    unpackers.push_back(std::move(unpacker));
    helpers  .push_back(std::move(  helper));
}

void ObjectExtraExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto helper = helpers[id].get();
    auto packer = packers[id].get();
    auto ov = objects[id];

    packer->update(ov->local(), stream);

    const auto& offsets = entangledHaloExchanger->getSendOffsets(id);

    for (int i = 0; i < helper->nBuffers; ++i)
        helper->send.sizes[i] = offsets[i+1] - offsets[i];
}

void ObjectExtraExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov     = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();
    const auto& map = entangledHaloExchanger->getMap(id);

    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf();

    const int nthreads = 256;
    mpark::visit([&](auto packerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            ObjectHaloExtraExchangerKernels::pack,
            map.size(), nthreads, 0, stream,
            ov->state->domain, packerHandler, map.devPtr(),
            helper->wrapSendData() );
    }, ExchangersCommon::getHandler(packer));
}

void ObjectExtraExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov       = objects[id];
    auto hov      = ov->halo();
    auto helper   = helpers[id].get();
    auto unpacker = unpackers[id].get();

    const auto& offsets = helper->recv.offsets;
    
    const int totalRecvd = offsets[helper->nBuffers];

    hov->resize_anew(totalRecvd * ov->objSize);
    unpacker->update(hov, stream);

    const int nthreads = 256;
    const int nblocks  = totalRecvd;
    const size_t shMemSize = offsets.size() * sizeof(offsets[0]);

    mpark::visit([&](auto unpackerHandler)
    {
        SAFE_KERNEL_LAUNCH(
            ObjectHaloExtraExchangerKernels::unpack,
            nblocks, nthreads, shMemSize, stream,
            helper->wrapRecvData(), unpackerHandler );
    }, ExchangersCommon::getHandler(unpacker));
}

} // namespace mirheo
