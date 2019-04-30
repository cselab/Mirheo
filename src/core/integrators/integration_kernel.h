#pragma once

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace IntegrationKernels
{

/**
 * \code transform(Particle& p, const float3 f, const float invm, const float dt) \endcode
 *  is a callable that performs integration. It is called for
 *  every particle and should change velocity and coordinate
 *  of the Particle according to the chosen integration scheme.
 *
 * Will read positions from \c oldPositions channel and write to positions
 * Will read velocities from velocities and write to velocities
 */
template<typename Transform>
__global__ void integrate(PVviewWithOldParticles pvView, const float dt, Transform transform)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= pvView.size) return;

    float4 pos = readNoCache(pvView.oldPositions + pid);
    float4 vel = readNoCache(pvView.velocities   + pid);
    Float3_int frc(pvView.forces[pid]);

    Particle p(pos, vel);

    transform(p, frc.v, pvView.invMass, dt);

    writeNoCache(pvView.positions  + pid, p.r2Float4());
    writeNoCache(pvView.velocities + pid, p.u2Float4());
}

} // namespace IntegrationKernels


template<typename Transform>
static void integrate(ParticleVector *pv, float dt, Transform transform, cudaStream_t stream)
{
    int nthreads = 128;

    // New particles now become old
    std::swap(pv->local()->positions(), *pv->local()->extraPerParticle.getData<float4>(ChannelNames::oldPositions));
    PVviewWithOldParticles pvView(pv, pv->local());

    SAFE_KERNEL_LAUNCH(
        IntegrationKernels::integrate,
        getNblocks(pvView.size, nthreads), nthreads, 0, stream,
        pvView, dt, transform );
}
