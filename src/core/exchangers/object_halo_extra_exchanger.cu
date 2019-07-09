#include "object_halo_extra_exchanger.h"
#include "object_halo_exchanger.h"
#include "exchange_helpers.h"

#include <core/logger.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/packers/objects.h>
#include <core/utils/kernel_launch.h>

namespace ObjectHaloExtraExchangerKernels
{
__global__ void pack(ObjectPackerHandler packer, const MapEntry *map,
                     BufferOffsetsSizesWrap dataWrap)
{
    const int tid         = threadIdx.x;
    const int objId       = blockIdx.x;
    const int numElements = gridDim.x;
    const int objSize     = packer.objSize;

    auto mapEntry = map[objId];

    const int bufId    = mapEntry.getBufId();
    const int srcObjId = mapEntry.getId();
    const int dstObjId = objId - dataWrap.offsets[bufId];
    
    auto buffer = dataWrap.getBuffer(bufId);

    size_t offsetBytes = 0;
    
    for (int pid = tid; pid < objSize; pid += blockDim.x)
    {
        int srcId = srcObjId * objSize + pid;
        int dstId = dstObjId * objSize + pid;

        offsetBytes = packer.particles.pack(srcId, dstId, buffer,
                                            numElements * objSize);
    }

    buffer += offsetBytes;
    if (tid == 0)
        packer.objects.pack(srcObjId, dstObjId, buffer, numElements);
}

__global__ void unpack(BufferOffsetsSizesWrap dataWrap, ObjectPackerHandler packer)
{
    const int objId = blockIdx.x;
    const int tid   = threadIdx.x;
    const int objSize = packer.objSize;

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
    
    size_t offsetBytes = 0;
    
    for (int pid = tid; pid < objSize; pid += blockDim.x)
    {
        const int dstPid = dstObjId * objSize + pid;
        const int srcPid = srcObjId * objSize + pid;
        offsetBytes = packer.particles.unpack(srcPid, dstPid, buffer,
                                              numElements * objSize);
    }

    buffer += offsetBytes;
    
    if (tid == 0)
        packer.objects.unpack(srcObjId, dstObjId, buffer, numElements);
}
} // namespace ObjectHaloExtraExchangerKernels


ObjectExtraExchanger::ObjectExtraExchanger(ObjectHaloExchanger *entangledHaloExchanger) :
    entangledHaloExchanger(entangledHaloExchanger)
{}

ObjectExtraExchanger::~ObjectExtraExchanger() = default;

bool ObjectExtraExchanger::needExchange(int id)
{
    return true;
}

void ObjectExtraExchanger::attach(ObjectVector *ov, const std::vector<std::string>& extraChannelNames)
{
    int id = objects.size();
    objects.push_back(ov);

    PackPredicate predicate = [extraChannelNames](const DataManager::NamedChannelDesc& namedDesc)
    {
        return std::find(extraChannelNames.begin(),
                         extraChannelNames.end(),
                         namedDesc.first)
            != extraChannelNames.end();
    };
    
    auto   packer = std::make_unique<ObjectPacker>(predicate);
    auto unpacker = std::make_unique<ObjectPacker>(predicate);
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
    
    SAFE_KERNEL_LAUNCH(
        ObjectHaloExtraExchangerKernels::pack,
        map.size(), nthreads, 0, stream,
        packer->handler(), map.devPtr(),
        helper->wrapSendData() );
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
        
    SAFE_KERNEL_LAUNCH(
        ObjectHaloExtraExchangerKernels::unpack,
        nblocks, nthreads, shMemSize, stream,
        helper->wrapRecvData(), unpacker->handler() );
}
