// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "operations.h"
#include "utils.h"

#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/utils/quaternion.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace rigid_operations_kernels
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
    const real3 com = make_real3( ovView.motions[objId].r );

    // Find the total force and torque
#pragma unroll 3
    for (int i = tid; i < ovView.objSize; i += blockDim.x)
    {
        const int offset = (objId * ovView.objSize + i);

        const real3 frc = make_real3(ovView.forces[offset]);
        const real3 r   = make_real3(ovView.positions[offset]) - com;

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
template <rigid_operations::ApplyTo action>
__global__ void applyRigidMotion(ROVview _ovView, const real4 *initialPositions)
{
    // NVCC 10.1 has a bug where, in debug mode, accessing member functions of
    // a grandparent class causes a crash in certain cases. Copying the
    // argument into a local variable circumvents the bug.
    ROVview ovView = _ovView;
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    const int objId = pid / ovView.objSize;
    const int locId = pid % ovView.objSize;

    if (pid >= ovView.nObjects*ovView.objSize) return;

    const auto motion = toRealMotion(ovView.motions[objId]);

    Particle p;
    ovView.readPosition(p, pid);

    p.r = motion.r + motion.q.rotate( make_real3(initialPositions[locId]) );

    if (action == rigid_operations::ApplyTo::PositionsAndVelocities)
    {
        ovView.readVelocity(p, pid);
        p.u = motion.vel + cross(motion.omega, p.r - motion.r);
        ovView.writeParticle(pid, p);
    }
    else
    {
        ovView.writePosition(pid, p.r2Real4());
    }
}

__global__ void clearRigidForcesFromMotions(ROVview ovView)
{
    const int objId = threadIdx.x + blockDim.x * blockIdx.x;
    if (objId >= ovView.nObjects) return;

    ovView.motions[objId].force  = {0,0,0};
    ovView.motions[objId].torque = {0,0,0};
}

} // namespace rigid_operations_kernels

namespace rigid_operations
{

void collectRigidForces(const ROVview& view, cudaStream_t stream)
{
    constexpr int nthreads = 128;
    const int nblocks = view.nObjects;

    SAFE_KERNEL_LAUNCH(
        rigid_operations_kernels::collectRigidForces,
        nblocks, nthreads, 0, stream,
        view );
}

void applyRigidMotion(const ROVview& view, const PinnedBuffer<real4>& initialPositions,
                      ApplyTo action, cudaStream_t stream)
{
    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    switch(action)
    {
    case ApplyTo::PositionsOnly:
        SAFE_KERNEL_LAUNCH(
            rigid_operations_kernels::applyRigidMotion<ApplyTo::PositionsOnly>,
            nblocks, nthreads, 0, stream,
            view, initialPositions.devPtr() );
        break;
    case ApplyTo::PositionsAndVelocities:
        SAFE_KERNEL_LAUNCH(
            rigid_operations_kernels::applyRigidMotion<ApplyTo::PositionsAndVelocities>,
            nblocks, nthreads, 0, stream,
            view, initialPositions.devPtr() );
        break;
    default:
        /* none */
        break;
    };
}

void clearRigidForcesFromMotions(const ROVview& view, cudaStream_t stream)
{
    constexpr int nthreads = 64;
    const int nblocks = getNblocks(view.nObjects, nthreads);

    SAFE_KERNEL_LAUNCH(
        rigid_operations_kernels::clearRigidForcesFromMotions,
        nblocks, nthreads, 0, stream,
        view );
}

} // namespace rigid_operations

} // namespace mirheo
