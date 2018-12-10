#include "object_halo_exchanger.h"
#include "exchange_helpers.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/pvs/extra_data/packers.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

template<bool QUERY=false>
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
        int cx = 1, cy = 1, cz = 1;

        if (prop.low.x  < -0.5f*domain.localSize.x + rc) cx = 0;
        if (prop.low.y  < -0.5f*domain.localSize.y + rc) cy = 0;
        if (prop.low.z  < -0.5f*domain.localSize.z + rc) cz = 0;

        if (prop.high.x >  0.5f*domain.localSize.x - rc) cx = 2;
        if (prop.high.y >  0.5f*domain.localSize.y - rc) cy = 2;
        if (prop.high.z >  0.5f*domain.localSize.z - rc) cz = 2;

        for (int ix = min(cx, 1); ix <= max(cx, 1); ix++)
            for (int iy = min(cy, 1); iy <= max(cy, 1); iy++)
                for (int iz = min(cz, 1); iz <= max(cz, 1); iz++)
                {
                    if (ix == 1 && iy == 1 && iz == 1) continue;
                    const int bufId = (iz*3 + iy)*3 + ix;
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

        const int ix = bufId % 3;
        const int iy = (bufId / 3) % 3;
        const int iz = bufId / 9;
        const float3 shift{ domain.localSize.x*(ix-1),
                            domain.localSize.y*(iy-1),
                            domain.localSize.z*(iz-1) };

        __syncthreads();
        if (tid == 0)
            shDstObjId = atomicAdd(dataWrap.sizes + bufId, 1);

        if (QUERY) {
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

void ObjectHaloExchanger::attach(ObjectVector* ov, float rc)
{
    objects.push_back(ov);
    rcs.push_back(rc);
    ExchangeHelper* helper = new ExchangeHelper(ov->name);
    helpers.push_back(helper);

    origins.push_back(new PinnedBuffer<int>(ov->local()->size()));

    info("Object vector %s (rc %f) was attached to halo exchanger", ov->name.c_str(), rc);
}

void ObjectHaloExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto rc  = rcs[id];
    auto helper = helpers[id];
    auto origin = origins[id];

    ov->findExtentAndCOM(stream, ParticleVectorType::Local);

    debug2("Counting halo objects of '%s'", ov->name.c_str());

    OVview ovView(ov, ov->local());
    ObjectPacker packer(ov, ov->local(), stream);
    helper->setDatumSize(packer.totalPackedSize_byte);

    helper->sendSizes.clear(stream);
    if (ovView.nObjects > 0)
    {
        const int nthreads = 256;

        SAFE_KERNEL_LAUNCH(
                getObjectHalos<true>,
                ovView.nObjects, nthreads, 0, stream,
                ov->domain, ovView, packer, rc, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }
}

void ObjectHaloExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov  = objects[id];
    auto rc  = rcs[id];
    auto helper = helpers[id];
    auto origin = origins[id];

    debug2("Downloading %d halo objects of '%s'", helper->sendOffsets[27], ov->name.c_str());

    OVview ovView(ov, ov->local());
    ObjectPacker packer(ov, ov->local(), stream);
    helper->setDatumSize(packer.totalPackedSize_byte);

    if (ovView.nObjects > 0)
    {
        // 1 int per particle: #objects x objSize x int
        origin->resize_anew(helper->sendOffsets[helper->nBuffers] * ovView.objSize);

        const int nthreads = 256;

        helper->resizeSendBuf();
        helper->sendSizes.clearDevice(stream);
        SAFE_KERNEL_LAUNCH(
                getObjectHalos<false>,
                ovView.nObjects, nthreads, 0, stream,
                ov->domain, ovView, packer, rc, helper->wrapSendData(), origin->devPtr() );
    }
}

void ObjectHaloExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id];

    int totalRecvd = helper->recvOffsets[helper->nBuffers];

    ov->halo()->resize_anew(totalRecvd * ov->objSize);
    OVview ovView(ov, ov->halo());
    ObjectPacker packer(ov, ov->halo(), stream);

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




