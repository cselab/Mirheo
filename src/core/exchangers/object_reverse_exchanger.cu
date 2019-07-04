#include "object_reverse_exchanger.h"
#include "exchange_helpers.h"
#include "object_halo_exchanger.h"

#include <core/logger.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/packers/objects.h>
#include <core/pvs/views/ov.h>
#include <core/utils/kernel_launch.h>

namespace ObjectReverseExchangerKernels
{

__global__ void reversePack(char *buffer, int startDstObjId,
                            OVview view, ObjectPackerHandler packer)
{
    const int objId = blockIdx.x;
    const int tid   = threadIdx.x;
    const int numElements = gridDim.x;

    const int dstObjId = objId;
    const int srcObjId = objId + startDstObjId;
    
    size_t offsetBytes = 0;
    
    for (int pid = tid; pid < view.objSize; pid += blockDim.x)
    {
        const int dstPid = dstObjId * view.objSize + pid;
        const int srcPid = srcObjId * view.objSize + pid;
        offsetBytes = packer.particles.pack(srcPid, dstPid, buffer,
                                            numElements * view.objSize);
    }

    buffer += offsetBytes;
    
    if (tid == 0)
        packer.objects.pack(srcObjId, dstObjId, buffer, numElements);
}

__global__ void reverseUnpackAndAdd(const OVview view, ObjectPackerHandler packer,
                                    const MapEntry *map, BufferOffsetsSizesWrap dataWrap)
{
    constexpr float eps = 1e-6f;
    int tid         = threadIdx.x;
    int srcObjId    = blockIdx.x;
    int numElements = gridDim.x;

    auto mapEntry = map[srcObjId];

    const int bufId    = mapEntry.getBufId();
    const int dstObjId = mapEntry.getId();
    
    auto buffer = dataWrap.buffer + dataWrap.offsetsBytes[bufId];

    size_t offsetBytes = 0;
    
    for (int pid = tid; pid < view.objSize; pid += blockDim.x)
    {
        int srcId = srcObjId * view.objSize + pid;
        int dstId = dstObjId * view.objSize + pid;

        offsetBytes = packer.particles.
            unpackAtomicAddNonZero(srcId, dstId, buffer,
                                   numElements * view.objSize, eps);
    }

    buffer += offsetBytes;
    if (tid == 0)
        packer.objects.unpackAtomicAddNonZero(srcObjId, dstObjId, buffer, numElements, eps);    
}

} // namespace ObjectReverseExchangerKernels


ObjectReverseExchanger::ObjectReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger) :
    entangledHaloExchanger(entangledHaloExchanger)
{}

ObjectReverseExchanger::~ObjectReverseExchanger() = default;

void ObjectReverseExchanger::attach(ObjectVector *ov, std::vector<std::string> channelNames)
{
    int id = objects.size();
    objects.push_back(ov);

    PackPredicate predicate = [channelNames](const DataManager::NamedChannelDesc& namedDesc)
    {
        return std::find(channelNames.begin(),
                         channelNames.end(),
                         namedDesc.first)
            != channelNames.end();
    };

    auto   packer = std::make_unique<ObjectPacker>(predicate);
    auto unpacker = std::make_unique<ObjectPacker>(predicate);
    auto   helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());
    
    packers  .push_back(std::move(  packer));
    unpackers.push_back(std::move(unpacker));
    helpers  .push_back(std::move(  helper));
}

bool ObjectReverseExchanger::needExchange(int id)
{
    return true;
}

void ObjectReverseExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto  helper  = helpers[id].get();
    auto& offsets = entangledHaloExchanger->getRecvOffsets(id);
    
    for (int i = 0; i < helper->nBuffers; i++)
        helper->send.sizes[i] = offsets[i+1] - offsets[i];
}

void ObjectReverseExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov     = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();
    
    debug2("Preparing '%s' data to reverse send", ov->name.c_str());

    packer->update(ov->halo(), stream);

    helper->computeSendOffsets();
    helper->send.uploadInfosToDevice(stream);
    helper->resizeSendBuf();

    OVview ovView(ov, ov->halo());
    
    for (int bufId = 0; bufId < helper->nBuffers; ++bufId)
    {
        int nObjs = helper->recv.sizes[bufId];

        if (bufId == helper->bulkId || nObjs == 0) continue;

        const int nthreads = 256;
        
        SAFE_KERNEL_LAUNCH(
            ObjectReverseExchangerKernels::reversePack,
            nObjs, nthreads, 0, stream,
            helper->send.buffer.devPtr() + helper->send.offsetsBytes[bufId],
            helper->send.offsets[bufId],
            ovView, packer->handler() );
    }
    
    debug2("Will send back data for %d objects", helper->send.offsets[helper->nBuffers]);
}

void ObjectReverseExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov       =   objects[id];
    auto helper   =   helpers[id].get();
    auto unpacker = unpackers[id].get();

    OVview ovView(ov, ov->local());

    unpacker->update(ov->local(), stream);
    
    int totalRecvd = helper->recv.offsets[helper->nBuffers];
    auto& map = entangledHaloExchanger->getMap(id);
    
    debug("Updating data for %d '%s' objects", totalRecvd, ov->name.c_str());

    const int nthreads = 256;
        
    SAFE_KERNEL_LAUNCH(
        ObjectReverseExchangerKernels::reverseUnpackAndAdd,
        map.size(), nthreads, 0, stream,
        ovView, unpacker->handler(),
        map.devPtr(), helper->wrapRecvData());
}
