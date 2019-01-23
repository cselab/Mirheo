#include "object_forces_reverse_exchanger.h"
#include "exchange_helpers.h"
#include "object_halo_exchanger.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/views/rov.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>


__device__ inline void atomicAddNonZero(float4* dest, float3 v)
{
    const float tol = 1e-7;

    float* fdest = (float*)dest;
    if (fabs(v.x) > tol) atomicAdd(fdest,     v.x);
    if (fabs(v.y) > tol) atomicAdd(fdest + 1, v.y);
    if (fabs(v.z) > tol) atomicAdd(fdest + 2, v.z);
}

__global__ void addHaloForces(
        const float4* recvForces, const int* origins,
        float4* forces, int objSize, int packedObjSize)
{
    const int objId = blockIdx.x;

    for (int pid = threadIdx.x; pid < objSize; pid += blockDim.x)
    {
        const int dstId = origins[objId*objSize + pid];
        Float3_int extraFrc( recvForces[objId*packedObjSize + pid] );
    
        atomicAddNonZero(forces + dstId, extraFrc.v);
    }
}

__global__ void addRigidForces(
        const float4* recvForces, const int nrecvd, const int* origins,
        ROVview view, int packedObjSize)
{
    const int gid = threadIdx.x + blockIdx.x*blockDim.x;
    const int objId = gid / 2;
    const int variant = gid % 2;
    if (objId >= nrecvd) return;

    const int dstObjId = origins[objId*view.objSize] / view.objSize;

    const float4* addr = recvForces + objId*packedObjSize + view.objSize;
    auto typedAddr = (const RigidReal4*) addr;

    RigidReal4 v = typedAddr[variant];

    if (variant == 0)
        atomicAdd(&view.motions[dstObjId].force,  {v.x, v.y, v.z});

    if (variant == 1)
        atomicAdd(&view.motions[dstObjId].torque, {v.x, v.y, v.z});
}

__global__ void packRigidForces(ROVview view, float4* output, int packedObjSize)
{
    const int objId = blockIdx.x;

    for (int pid = threadIdx.x; pid < view.objSize; pid += blockDim.x)
        output[objId*packedObjSize + pid] = view.forces[objId*view.objSize + pid];

    float4* addr = output + objId*packedObjSize + view.objSize;
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


//===============================================================================================
// Member functions
//===============================================================================================

bool ObjectForcesReverseExchanger::needExchange(int id)
{
    return true;
}

void ObjectForcesReverseExchanger::attach(ObjectVector* ov)
{
    objects.push_back(ov);

    int psize = ov->objSize;
    if (dynamic_cast<RigidObjectVector*>(ov) != 0)
        psize += 2 * sizeof(RigidReal) / sizeof(float);

    auto helper = std::make_unique<ExchangeHelper>(ov->name);
    helper->setDatumSize(psize*sizeof(float4));
        
    helpers.push_back(std::move(helper));
}


void ObjectForcesReverseExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto helper = helpers[id].get();
    auto& offsets = entangledHaloExchanger->getRecvOffsets(id);

    for (int i=0; i < helper->nBuffers; i++)
        helper->sendSizes[i] = offsets[i+1] - offsets[i];
}

void ObjectForcesReverseExchanger::prepareData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();

    debug2("Preparing '%s' forces to sending back", ov->name.c_str());

    helper->computeSendOffsets();
    helper->resizeSendBuf();

    auto rov = dynamic_cast<RigidObjectVector*>(ov);
    if (rov != nullptr)
    {
        int psize = rov->objSize + 2 * sizeof(RigidReal) / sizeof(float);
        ROVview view(rov, rov->halo());

        const int nthreads = 128;
        SAFE_KERNEL_LAUNCH(
                packRigidForces,
                view.nObjects, nthreads, 0, stream,
                view, (float4*)helper->sendBuf.devPtr(), psize);

    }
    else
    {
        CUDA_Check( cudaMemcpyAsync( helper->sendBuf.devPtr(),
                                     ov->halo()->forces.devPtr(),
                                     helper->sendBuf.size(), cudaMemcpyDeviceToDevice, stream ) );
    }

    debug2("Will send back forces for %d objects", helper->sendOffsets[helper->nBuffers]);
}

void ObjectForcesReverseExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto ov = objects[id];
    auto helper = helpers[id].get();

    int totalRecvd = helper->recvOffsets[helper->nBuffers];
    auto& origins = entangledHaloExchanger->getOrigins(id);

    debug("Updating forces for %d %s objects", totalRecvd, ov->name.c_str());

    int psize = ov->objSize;
    auto rov = dynamic_cast<RigidObjectVector*>(ov);
    if (rov != nullptr) psize += 2 * sizeof(RigidReal) / sizeof(float);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            addHaloForces,
            totalRecvd, nthreads, 0, stream,
            (const float4*)helper->recvBuf.devPtr(),     /* source */
            (const int*)origins.devPtr(),                /* destination ids here */
            (float4*)ov->local()->forces.devPtr(),       /* add to */
            ov->objSize, psize );

    if (rov != nullptr)
    {
        ROVview view(rov, rov->local());
        SAFE_KERNEL_LAUNCH(
                addRigidForces,
                getNblocks(totalRecvd, nthreads), nthreads, 0, stream,
                (const float4*)helper->recvBuf.devPtr(),     /* source */
                totalRecvd,
                (const int*)origins.devPtr(),                /* destination ids here */
                view, psize );                               /* add to, packed size */
    }
}





