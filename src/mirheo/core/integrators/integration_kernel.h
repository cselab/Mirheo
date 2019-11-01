#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace IntegrationKernels
{

/**
 * \code transform(Particle& p, const real3 f, const real invm, const real dt) \endcode
 *  is a callable that performs integration. It is called for
 *  every particle and should change velocity and coordinate
 *  of the Particle according to the chosen integration scheme.
 *
 * Will read positions from \c oldPositions channel and write to positions
 * Will read velocities from velocities and write to velocities
 */
template<typename Transform>
__global__ void integrate(PVviewWithOldParticles pvView, const real dt, Transform transform)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= pvView.size) return;

    real4 pos = readNoCache(pvView.oldPositions + pid);
    real4 vel = readNoCache(pvView.velocities   + pid);
    Real3_int frc(pvView.forces[pid]);

    Particle p(pos, vel);

    transform(p, frc.v, pvView.invMass, dt);

    writeNoCache(pvView.positions  + pid, p.r2Real4());
    writeNoCache(pvView.velocities + pid, p.u2Real4());
}

} // namespace IntegrationKernels


template<typename Transform>
static void integrate(ParticleVector *pv, real dt, Transform transform, cudaStream_t stream)
{
    constexpr int nthreads = 128;

    // New particles now become old
    std::swap(pv->local()->positions(), *pv->local()->dataPerParticle.getData<real4>(ChannelNames::oldPositions));
    PVviewWithOldParticles pvView(pv, pv->local());

    SAFE_KERNEL_LAUNCH(
        IntegrationKernels::integrate,
        getNblocks(pvView.size, nthreads), nthreads, 0, stream,
        pvView, dt, transform );
}
