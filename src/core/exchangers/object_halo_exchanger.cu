#include "object_halo_exchanger.h"
#include "exchange_helpers.h"
#include "utils/fragments_mapping.h"
#include "packers/map.h"
#include "packers/objects.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
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
__global__ void getObjectHaloMap(const DomainInfo domain, const OVview view, MapEntry *map,
                                 const float rc, BufferOffsetsSizesWrap dataWrap)
{
    const int objId = threadIdx.x + blockIdx.x * blockDim.x;

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

    for (int i = 0; i < nHalos; ++i)
    {
        const int bufId = validHalos[i];

        int dstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

        if (packMode == PackMode::Query)
        {
            continue;
        }
        else
        {
            int myOffset  = dataWrap.offsets[bufId] + dstObjId;
            map[myOffset] = MapEntry(objId, bufId);
        }
    }
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
    
    auto packer = std::make_unique<ObjectsPacker>(ov, [channels](const DataManager::NamedChannelDesc& namedDesc) {
        return std::find(channels.begin(), channels.end(), namedDesc.first) != channels.end();
    });
    
    auto helper = std::make_unique<ExchangeHelper>(ov->name, id, packer.get());

    packers.push_back(std::move(packer));
    helpers.push_back(std::move(helper));
    
    info("Object vector %s (rc %f) was attached to halo exchanger", ov->name.c_str(), rc);
}

void ObjectHaloExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto rc  = rcs[id];
    auto helper = helpers[id].get();

    ov->findExtentAndCOM(stream, ParticleVectorType::Local);

    debug2("Counting halo objects of '%s'", ov->name.c_str());

    OVview ovView(ov, ov->local());
    helper->send.sizes.clear(stream);

    if (ovView.nObjects > 0)
    {
        const int nthreads = 32;

        SAFE_KERNEL_LAUNCH(
            ObjectHaloExchangeKernels::getObjectHaloMap<PackMode::Query>,
            getNblocks(ovView.nObjects, nthreads), nthreads, 0, stream,
            ov->state->domain, ovView, nullptr, rc, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }
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
        const int nthreads = 32;
        debug2("Downloading %d halo objects of '%s'", nhalo, ov->name.c_str());

        helper->resizeSendBuf();
        helper->send.sizes.clearDevice(stream);
        helper->map.resize_anew(nhalo);
        
        SAFE_KERNEL_LAUNCH(
            ObjectHaloExchangeKernels::getObjectHaloMap<PackMode::Pack>,
            getNblocks(ovView.nObjects, nthreads), nthreads, 0, stream,
            ov->state->domain, ovView, helper->map.devPtr(), rc, helper->wrapSendData());

        packer->packToBuffer(ov->local(), helper->map, &helper->send, stream);
    }
}

void ObjectHaloExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    int totalRecvd = helper->recv.offsets[helper->nBuffers];

    ov->halo()->resize_anew(totalRecvd * ov->objSize);

    packer->unpackFromBuffer(ov->halo(), &helper->recv, 0, stream);
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
