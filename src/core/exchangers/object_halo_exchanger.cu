#include "object_halo_exchanger.h"
#include "exchange_helpers.h"
#include "utils/fragments_mapping.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/packers/objects.h>
#include <core/pvs/views/ov.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

enum class PackMode
{
    Query, Pack
};

namespace ObjectHaloExchangeKernels
{

template <PackMode packMode>
__global__ void getObjectHaloAndMap(const DomainInfo domain, const OVview view, MapEntry *map,
                                    const float rc, ObjectPackerHandler packer,
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

        if (prop.low.x  < -0.5f*domain.localSize.x + rc) dx = -1;
        if (prop.low.y  < -0.5f*domain.localSize.y + rc) dy = -1;
        if (prop.low.z  < -0.5f*domain.localSize.z + rc) dz = -1;

        if (prop.high.x >  0.5f*domain.localSize.x - rc) dx = 1;
        if (prop.high.y >  0.5f*domain.localSize.y - rc) dy = 1;
        if (prop.high.z >  0.5f*domain.localSize.z - rc) dz = 1;

        for (int ix = min(dx, 0); ix <= max(dx, 0); ++ix)
            for (int iy = min(dy, 0); iy <= max(dy, 0); ++iy)
                for (int iz = min(dz, 0); iz <= max(dz, 0); ++iz)
                {
                    if (ix == 0 && iy == 0 && iz == 0) continue;
                    const int bufId = FragmentMapping::getId(ix, iy, iz);
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

            const int3 dir = FragmentMapping::getDir(bufId);
            
            const float3 shift{ - domain.localSize.x * dir.x,
                                - domain.localSize.y * dir.y,
                                - domain.localSize.z * dir.z };

            auto buffer = dataWrap.buffer + dataWrap.offsetsBytes[bufId];
            int numElements = dataWrap.offsets[bufId+1] - dataWrap.offsets[bufId];

            size_t offsetBytes = 0;

            // save data to buffer
            
            for (int pid = tid; pid < view.objSize; pid += blockDim.x)
            {
                const int srcPid = objId      * view.objSize + pid;
                const int dstPid = shDstObjId * view.objSize + pid;

                offsetBytes = packer.particles.packShift(srcPid, dstPid, buffer,
                                                         numElements * view.objSize,
                                                         shift);
            }

            buffer += offsetBytes;
            
            if (tid == 0)
                packer.objects.packShift(objId, shDstObjId, buffer, numElements, shift);

            // save map
            
            int myOffset = dataWrap.offsets[bufId] + shDstObjId;
            map[myOffset] = MapEntry(objId, bufId);
        }
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

} // namespace ObjectHaloExchangeKernels

//===============================================================================================
// Member functions
//===============================================================================================

bool ObjectHaloExchanger::needExchange(int id)
{
    return !objects[id]->haloValid;
}

ObjectHaloExchanger::ObjectHaloExchanger() = default;
ObjectHaloExchanger::~ObjectHaloExchanger() = default;

void ObjectHaloExchanger::attach(ObjectVector *ov, float rc, const std::vector<std::string>& extraChannelNames)
{
    int id = objects.size();
    objects.push_back(ov);
    rcs.push_back(rc);

    auto channels = extraChannelNames;
    channels.push_back(ChannelNames::positions);
    channels.push_back(ChannelNames::velocities);

    PackPredicate predicate = [channels](const DataManager::NamedChannelDesc& namedDesc)
    {
        return std::find(channels.begin(), channels.end(), namedDesc.first) != channels.end();
    };
    
    auto   packer = std::make_unique<ObjectPacker>(predicate);
    auto unpacker = std::make_unique<ObjectPacker>(predicate);
    auto   helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());

    packers  .push_back(std::move(  packer));
    unpackers.push_back(std::move(unpacker));
    helpers  .push_back(std::move(  helper));
    
    info("Object vector %s (rc %f) was attached to halo exchanger", ov->name.c_str(), rc);
}

void ObjectHaloExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto rc  = rcs[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    ov->findExtentAndCOM(stream, ParticleVectorType::Local);

    debug2("Counting halo objects of '%s'", ov->name.c_str());

    OVview ovView(ov, ov->local());
    helper->send.sizes.clear(stream);
    packer->update(ov->local(), stream);

    if (ovView.nObjects > 0)
    {
        const int nthreads = 256;

        SAFE_KERNEL_LAUNCH(
            ObjectHaloExchangeKernels::getObjectHaloAndMap<PackMode::Query>,
            ovView.nObjects, nthreads, 0, stream,
            ov->state->domain, ovView, nullptr, rc,
            packer->handler(), helper->wrapSendData() );
    }

    helper->computeSendOffsets_Dev2Dev(stream);
}

void ObjectHaloExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto rc  = rcs[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    int nhalo = helper->send.offsets[helper->nBuffers];
    OVview ovView(ov, ov->local());

    if (ovView.nObjects > 0)
    {
        const int nthreads = 256;
        debug2("Downloading %d halo objects of '%s'", nhalo, ov->name.c_str());

        helper->resizeSendBuf();
        helper->send.sizes.clearDevice(stream);
        helper->map.resize_anew(nhalo);
        
        SAFE_KERNEL_LAUNCH(
            ObjectHaloExchangeKernels::getObjectHaloAndMap<PackMode::Pack>,
            ovView.nObjects, nthreads, 0, stream,
            ov->state->domain, ovView, helper->map.devPtr(), rc,
            packer->handler(), helper->wrapSendData());
    }
}

void ObjectHaloExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto unpacker = unpackers[id].get();

    int totalRecvd = helper->recv.offsets[helper->nBuffers];

    ov->halo()->resize_anew(totalRecvd * ov->objSize);
    OVview ovView(ov, ov->halo());
    
    unpacker->update(ov->halo(), stream);

    for (int bufId = 0; bufId < helper->nBuffers; ++bufId)
    {
        int nObjs = helper->recv.sizes[bufId];

        if (bufId == helper->bulkId || nObjs == 0) continue;

        const int nthreads = 256;
        
        SAFE_KERNEL_LAUNCH(
            ObjectHaloExchangeKernels::unpackObjects,
            nObjs, nthreads, 0, stream,
            helper->recv.buffer.devPtr() + helper->recv.offsetsBytes[bufId],
            helper->recv.offsets[bufId],
            ovView, unpacker->handler() );
    }
}

PinnedBuffer<int>& ObjectHaloExchanger::getSendOffsets(int id)
{
    return helpers[id]->send.offsets;
}

PinnedBuffer<int>& ObjectHaloExchanger::getRecvOffsets(int id)
{
    return helpers[id]->recv.offsets;
}

DeviceBuffer<MapEntry>& ObjectHaloExchanger::getMap(int id)
{
    return helpers[id]->map;
}
