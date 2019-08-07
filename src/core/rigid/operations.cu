#include "operations.h"
#include "utils.h"

#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/views/rov.h>
#include <core/utils/quaternion.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace RigidOperationsKernels
{

/**
 * Find total force and torque on objects, write it to motions
 */
__global__ void collectRigidForces(ROVview ovView)
{
    const int objId = blockIdx.x;
    const int tid = threadIdx.x;
    if (objId >= ovView.nObjects) return;

    RigidReal3 force {0,0,0};
    RigidReal3 torque{0,0,0};
    float3 com = make_float3( ovView.motions[objId].r );

    // Find the total force and torque
#pragma unroll 3
    for (int i = tid; i < ovView.objSize; i += blockDim.x)
    {
        const int offset = (objId * ovView.objSize + i);

        const float3 frc = make_float3(ovView.forces[offset]);
        const float3 r   = make_float3(ovView.readPosition(offset)) - com;

        force  += frc;
        torque += cross(r, frc);
    }

    force  = warpReduce( force,  [] (RigidReal a, RigidReal b) { return a+b; } );
    torque = warpReduce( torque, [] (RigidReal a, RigidReal b) { return a+b; } );

    if ( tid % warpSize == 0 )
    {
        atomicAdd(&ovView.motions[objId].force,  force);
        atomicAdd(&ovView.motions[objId].torque, torque);
    }
}

/**
 * Rotates and translates the initial positions according to new position and orientation
 * compute also velocity if template parameter set to corresponding value
 */
template <RigidOperations::ApplyTo action>
__global__ void applyRigidMotion(ROVview ovView, const float4 *initialPositions)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const int objId = pid / ovView.objSize;
    const int locId = pid % ovView.objSize;

    if (pid >= ovView.nObjects*ovView.objSize) return;

    const auto motion = toSingleMotion(ovView.motions[objId]);

    Particle p;
    ovView.readPosition(p, pid);

    // Some explicit conversions for double precision
    p.r = motion.r + rotate( make_float3(initialPositions[locId]), motion.q );

    if (action == RigidOperations::ApplyTo::PositionsAndVelocities)
    {
        ovView.readVelocity(p, pid);
        p.u = motion.vel + cross(motion.omega, p.r - motion.r);
        ovView.writeParticle(pid, p);
    }
    else
    {
        ovView.writePosition(pid, p.r2Float4());
    }
}

__global__ void clearRigidForces(ROVview ovView)
{
    const int objId = threadIdx.x + blockDim.x * blockIdx.x;
    if (objId >= ovView.nObjects) return;

    ovView.motions[objId].force  = {0,0,0};
    ovView.motions[objId].torque = {0,0,0};
}

} // namespace RigidOperationsKernels

namespace RigidOperations
{

void collectRigidForces(const ROVview& view, cudaStream_t stream)
{
    constexpr int nthreads = 128;
    const int nblocks = view.nObjects;

    SAFE_KERNEL_LAUNCH(
        RigidOperationsKernels::collectRigidForces,
        nblocks, nthreads, 0, stream,
        view );
}

void applyRigidMotion(const ROVview& view, const PinnedBuffer<float4>& initialPositions,
                      ApplyTo action, cudaStream_t stream)
{
    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    switch(action)
    {
    case ApplyTo::PositionsOnly:
        SAFE_KERNEL_LAUNCH(
            RigidOperationsKernels::applyRigidMotion<ApplyTo::PositionsOnly>,
            nblocks, nthreads, 0, stream,
            view, initialPositions.devPtr() );
        break;
    case ApplyTo::PositionsAndVelocities:
        SAFE_KERNEL_LAUNCH(
            RigidOperationsKernels::applyRigidMotion<ApplyTo::PositionsAndVelocities>,
            nblocks, nthreads, 0, stream,
            view, initialPositions.devPtr() );
        break;
    default:
        /* none */
        break;
    };
}

void clearRigidForces(const ROVview& view, cudaStream_t stream)
{
    constexpr int nthreads = 64;
    const int nblocks = getNblocks(view.nObjects, nthreads);

    SAFE_KERNEL_LAUNCH(
        RigidOperationsKernels::clearRigidForces,
        nblocks, nthreads, 0, stream,
        view );
}

} // namespace RigidOperations
