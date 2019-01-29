#include "object_halo_exchanger.h"
#include "exchange_helpers.h"
#include "fragments_mapping.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/pvs/extra_data/packers.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

enum class PackMode
{
    Query, Pack
};

template <PackMode packMode>
__global__ void getObjectHalos(const DomainInfo domain, const OVview view, const ObjectPacker packer,
        const float rc, BufferOffsetsSizesWrap dataWrap, int* haloParticleIds = nullptr)
{
    const int objId = blockIdx.x;
    const int tid = threadIdx.x;

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

        for (int ix = min(dx, 0); ix <= max(dx, 0); ix++)
            for (int iy = min(dy, 0); iy <= max(dy, 0); iy++)
                for (int iz = min(dz, 0); iz <= max(dz, 0); iz++)
                {
                    if (ix == 0 && iy == 0 && iz == 0) continue;
                    const int bufId = FragmentMapping::getId(ix, iy, iz);
                    validHalos[nHalos] = bufId;
                    nHalos++;
                }
    }

    // Copy objects to each halo
    // TODO: maybe other loop order?
    __shared__ int shDstObjId;
    for (int i=0; i<nHalos; i++)
    {
        const int bufId = validHalos[i];

        const int3 dir = FragmentMapping::getDir(bufId);
        
        const float3 shift{ domain.localSize.x * dir.x,
                            domain.localSize.y * dir.y,
                            domain.localSize.z * dir.z };

        __syncthreads();
        if (tid == 0)
            shDstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

        if (packMode == PackMode::Query) {
            continue;
        }
        else {
            __syncthreads();

            int myOffset = dataWrap.offsets[bufId] + shDstObjId;
            int* partIdsAddr = haloParticleIds + view.objSize * myOffset;

            // Save particle origins
            for (int pid = tid; pid < view.objSize; pid += blockDim.x)
            {
                const int srcId = objId * view.objSize + pid;
                partIdsAddr[pid] = srcId;
            }

            char* dstAddr = dataWrap.buffer + packer.totalPackedSize_byte * myOffset;
            for (int pid = tid; pid < view.objSize; pid += blockDim.x)
            {
                const int srcPid = objId * view.objSize + pid;
                packer.part.packShift(srcPid, dstAddr + pid*packer.part.packedSize_byte, -shift);
            }

            dstAddr += view.objSize * packer.part.packedSize_byte;
            if (tid == 0) packer.obj.packShift(objId, dstAddr, -shift);
        }
    }
}

__global__ static void unpackObject(const char* from, const int startDstObjId, OVview view, ObjectPacker packer)
{
    const int objId = blockIdx.x;
    const int tid = threadIdx.x;

    const char* srcAddr = from + packer.totalPackedSize_byte * objId;

    for (int pid = tid; pid < view.objSize; pid += blockDim.x)
    {
        const int dstId = (startDstObjId+objId)*view.objSize + pid;
        packer.part.unpack(srcAddr + pid*packer.part.packedSize_byte, dstId);
    }

    srcAddr += view.objSize * packer.part.packedSize_byte;
    if (tid == 0) packer.obj.unpack(srcAddr, startDstObjId+objId);
}

//===============================================================================================
// Member functions
//===============================================================================================

bool ObjectHaloExchanger::needExchange(int id)
{
    return !objects[id]->haloValid;
}

void ObjectHaloExchanger::attach(ObjectVector* ov, float rc, const std::vector<std::string>& extraChannelNames)
{
    int id = objects.size();
    objects.push_back(ov);
    rcs.push_back(rc);

    auto helper = std::make_unique<ExchangeHelper>(ov->name, id);
    helpers.push_back(std::move(helper));

    auto origin = std::make_unique<PinnedBuffer<int>>(ov->local()->size());    
    origins.push_back(std::move(origin));

    packPredicates.push_back([extraChannelNames](const ExtraDataManager::NamedChannelDesc& namedDesc) {
        bool needExchange = namedDesc.second->communication == ExtraDataManager::CommunicationMode::NeedExchange;
        bool isRequired   = std::find(extraChannelNames.begin(), extraChannelNames.end(), namedDesc.first) != extraChannelNames.end();
        return needExchange || isRequired;
    });
    
    info("Object vector %s (rc %f) was attached to halo exchanger", ov->name.c_str(), rc);
}

void ObjectHaloExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto rc  = rcs[id];
    auto helper = helpers[id].get();
    auto origin = origins[id].get();

    ov->findExtentAndCOM(stream, ParticleVectorType::Local);

    debug2("Counting halo objects of '%s'", ov->name.c_str());

    OVview ovView(ov, ov->local());
    ObjectPacker packer(ov, ov->local(), packPredicates[id], stream);
    helper->setDatumSize(packer.totalPackedSize_byte);

    helper->sendSizes.clear(stream);
    if (ovView.nObjects > 0)
    {
        const int nthreads = 256;

        SAFE_KERNEL_LAUNCH(
                getObjectHalos<PackMode::Query>,
                ovView.nObjects, nthreads, 0, stream,
                ov->state->domain, ovView, packer, rc, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }
}

void ObjectHaloExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto rc  = rcs[id];
    auto helper = helpers[id].get();
    auto origin = origins[id].get();

    debug2("Downloading %d halo objects of '%s'",
           helper->sendOffsets[FragmentMapping::numFragments], ov->name.c_str());

    OVview ovView(ov, ov->local());
    ObjectPacker packer(ov, ov->local(), packPredicates[id], stream);
    helper->setDatumSize(packer.totalPackedSize_byte);

    if (ovView.nObjects > 0)
    {
        // 1 int per particle: #objects x objSize x int
        origin->resize_anew(helper->sendOffsets[helper->nBuffers] * ovView.objSize);

        const int nthreads = 256;

        helper->resizeSendBuf();
        helper->sendSizes.clearDevice(stream);
        SAFE_KERNEL_LAUNCH(
                getObjectHalos<PackMode::Pack>,
                ovView.nObjects, nthreads, 0, stream,
                ov->state->domain, ovView, packer, rc, helper->wrapSendData(), origin->devPtr() );
    }
}

void ObjectHaloExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();

    int totalRecvd = helper->recvOffsets[helper->nBuffers];

    ov->halo()->resize_anew(totalRecvd * ov->objSize);
    OVview ovView(ov, ov->halo());
    ObjectPacker packer(ov, ov->halo(), packPredicates[id], stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            unpackObject,
            totalRecvd, nthreads, 0, stream,
            helper->recvBuf.devPtr(), 0, ovView, packer );
}

PinnedBuffer<int>& ObjectHaloExchanger::getRecvOffsets(int id)
{
    return helpers[id]->recvOffsets;
}

PinnedBuffer<int>& ObjectHaloExchanger::getOrigins(int id)
{
    return *origins[id];
}

ObjectHaloExchanger::~ObjectHaloExchanger() = default;



