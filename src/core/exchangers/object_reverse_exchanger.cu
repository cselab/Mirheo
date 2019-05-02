#include "object_reverse_exchanger.h"
#include "exchange_helpers.h"
#include "object_halo_exchanger.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/views/rov.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

namespace ObjectReverseExchangerKernels
{

__global__ void packObjectForces(OVview view, char *output, int datumSize)
{
    const int objId = blockIdx.x;
    float4 *addr = (float4*) (output + objId * datumSize);
    
    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
        addr[pid] = view.forces[objId*view.objSize + pid];
}

__global__ void packRigidForces(ROVview view, char *output, int datumSize)
{
    const int objId = blockIdx.x;

    float4 *addr = (float4*) (output + objId * datumSize);
    
    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
        addr[pid] = view.forces[objId*view.objSize + pid];

    addr += view.objSize;
    auto typedAddr = (RigidReal4*) addr;

    if (threadIdx.x == 0)
    {
        auto f = view.motions[objId].force;

        typedAddr[0] = {f.x, f.y, f.z, (RigidReal)0};
    }

    if (threadIdx.x == 1)
    {
        auto t = view.motions[objId].torque;

        typedAddr[1] = {t.x, t.y, t.z, (RigidReal)0};
    }
}

__global__ void packExtraData(int objSize, int offsetPerObject, int byteSizePerObject, ParticleExtraPacker packer, char *dstBuffer)
{
    int objId = blockIdx.x;
    int tid = threadIdx.x;

    char *dstAddr = dstBuffer + byteSizePerObject * objId + offsetPerObject;

    for (int pid = tid; pid < objSize; pid += blockDim.x)
    {
        int srcId = objId * objSize + pid;
        packer.pack(srcId, dstAddr + pid * packer.packedSize_byte);
    }
}


__device__ inline void atomicAddNonZero(float4 *dest, float3 v)
{
    const float tol = 1e-7;

    float* fdest = (float*)dest;
    if (fabs(v.x) > tol) atomicAdd(fdest,     v.x);
    if (fabs(v.y) > tol) atomicAdd(fdest + 1, v.y);
    if (fabs(v.z) > tol) atomicAdd(fdest + 2, v.z);
}

__global__ void addHaloForces(const char *recvBuffer, const int *origins,
                              float4 *forces, int objSize, int datumSize)
{
    const int objId = blockIdx.x;
    const float4 *recvForces = (const float4*) (recvBuffer + objId * datumSize);
    
    for (int pid = threadIdx.x; pid < objSize; pid += blockDim.x)
    {
        const int dstId = origins[objId*objSize + pid];
        Float3_int extraFrc( recvForces[pid] );
    
        atomicAddNonZero(forces + dstId, extraFrc.v);
    }
}

__global__ void addRigidForces(const char *recvBuffer, int nrecvd, const int *origins,
                               ROVview view, int datumSize)
{
    const int gid = threadIdx.x + blockIdx.x*blockDim.x;
    const int objId = gid / 2;
    const int variant = gid % 2;
    if (objId >= nrecvd) return;

    const int dstObjId = origins[objId*view.objSize] / view.objSize;

    const float4 *recvForces = (const float4*) (recvBuffer + objId * datumSize);
    const float4 *addr = recvForces + view.objSize;
    auto typedAddr = (const RigidReal4*) addr;

    RigidReal4 v = typedAddr[variant];

    if (variant == 0)
        atomicAdd(&view.motions[dstObjId].force,  {v.x, v.y, v.z});

    if (variant == 1)
        atomicAdd(&view.motions[dstObjId].torque, {v.x, v.y, v.z});
}

__global__ void unpackAndAddExtraData(int objSize, int offsetPerObject, int byteSizePerObject, const int *origins, const char *srcBuffer, ParticleExtraPacker packer)
{
    int objId = blockIdx.x;
    int tid = threadIdx.x;
    
    const char *srcAddr = srcBuffer + byteSizePerObject * objId + offsetPerObject;

    for (int pid = tid; pid < objSize; pid += blockDim.x)
    {
        int dstId = origins[objId * objSize + pid];
        packer.unpackAtomicAdd(srcAddr + pid * packer.packedSize_byte, dstId);
    }
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

    auto forcesIt = std::find(channelNames.begin(), channelNames.end(), ChannelNames::forces);
    bool needExchForces = forcesIt != channelNames.end();
    needForces.push_back(needExchForces);

    if (needExchForces)
        channelNames.erase(forcesIt); // forces are not extra data
    
    packPredicates.push_back([channelNames](const DataManager::NamedChannelDesc& namedDesc) {
        return std::find(channelNames.begin(), channelNames.end(), namedDesc.first) != channelNames.end();
    });

    ParticleExtraPacker packer(ov, ov->local(), packPredicates[id], defaultStream);
    int datumSize = getForceDatumSize(id);
    datumSize += ov->objSize * packer.packedSize_byte;
    
    auto helper = std::make_unique<ExchangeHelper>(ov->name, id);
    helper->setDatumSize(datumSize);
    
    helpers.push_back(std::move(helper));    
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
        helper->sendSizes[i] = offsets[i+1] - offsets[i];
}

void ObjectReverseExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto needExchForces = needForces[id];
    int objSize = ov->objSize;
    
    debug2("Preparing '%s' data to sending back with %sforces",
           ov->name.c_str(), needExchForces ? "" : "no ");

    ParticleExtraPacker packer(ov, ov->halo(), packPredicates[id], stream);

    auto rov = dynamic_cast<RigidObjectVector*>(ov);
    
    int forceDatumSize = getForceDatumSize(id);
    int datumSize = forceDatumSize + ov->objSize * packer.packedSize_byte;
    
    helper->setDatumSize(datumSize);
    helper->computeSendOffsets();
    helper->resizeSendBuf();

    const int nthreads = 128;
    int nObjects = ov->halo()->nObjects;

    if (needExchForces)
    {
        if (rov != nullptr)
        {
            ROVview view(rov, rov->halo());
            // pack particle forces + rigid motion force and torque
            SAFE_KERNEL_LAUNCH(
                ObjectReverseExchangerKernels::packRigidForces,
                view.nObjects, nthreads, 0, stream,
                view, helper->sendBuf.devPtr(), datumSize);
        }
        else
        {
            OVview view(ov, ov->halo());
            // pack particle forces only
            SAFE_KERNEL_LAUNCH(
                ObjectReverseExchangerKernels::packObjectForces,
                view.nObjects, nthreads, 0, stream,
                view, helper->sendBuf.devPtr(), datumSize);
        }
    }

    if (packer.packedSize_byte)
    {
        // pack extra data only
        SAFE_KERNEL_LAUNCH(
            ObjectReverseExchangerKernels::packExtraData,
            nObjects, nthreads, 0, stream,
            objSize, forceDatumSize, datumSize, packer, helper->sendBuf.devPtr() );
    }
    
    debug2("Will send back data for %d objects", helper->sendOffsets[helper->nBuffers]);
}

void ObjectReverseExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();
    auto needExchForces = needForces[id];
    int objSize = ov->objSize;
    
    int totalRecvd = helper->recvOffsets[helper->nBuffers];
    auto& origins = entangledHaloExchanger->getOrigins(id);

    debug("Updating data for %d '%s' objects", totalRecvd, ov->name.c_str());

    ParticleExtraPacker packer(ov, ov->local(), packPredicates[id], stream);
    int forceDatumSize = getForceDatumSize(id);
    int datumSize = forceDatumSize + ov->objSize * packer.packedSize_byte;    
    
    const int nthreads = 128;

    if (needExchForces)
    {
        auto rov = dynamic_cast<RigidObjectVector*>(ov);

        SAFE_KERNEL_LAUNCH(
            ObjectReverseExchangerKernels::addHaloForces,
            totalRecvd, nthreads, 0, stream,
            helper->recvBuf.devPtr(),                    /* source */
            origins.devPtr(),                            /* destination ids here */
            (float4*)ov->local()->forces().devPtr(),     /* add to */
            ov->objSize, datumSize );

            if (rov != nullptr)
            {
                ROVview view(rov, rov->local());
                SAFE_KERNEL_LAUNCH(
                    ObjectReverseExchangerKernels::addRigidForces,
                    getNblocks(totalRecvd, nthreads), nthreads, 0, stream,
                    helper->recvBuf.devPtr(),  /* source */
                    totalRecvd,
                    origins.devPtr(),          /* destination ids here */
                    view, datumSize );         /* add to, packed size */
            }
    }
    
    if (packer.packedSize_byte)
    {
        SAFE_KERNEL_LAUNCH(
            ObjectReverseExchangerKernels::unpackAndAddExtraData,
            totalRecvd, nthreads, 0, stream,
            objSize, forceDatumSize, datumSize,
            origins.devPtr(),            /* destination ids here */
            helper->recvBuf.devPtr(),    /* source */
            packer);
    }
    
}

int ObjectReverseExchanger::getForceDatumSize(int id) const
{    
    if (!needForces[id]) return 0;

    auto ov = objects[id];
    int objSize = ov->objSize;
    int forcesSize = sizeof(Force) * objSize;  // forces per particle

    if (dynamic_cast<RigidObjectVector*>(ov) != nullptr)
        forcesSize += 2 * sizeof(RigidReal4);  // force and torque per object

    return forcesSize;
}
