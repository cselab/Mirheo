// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "vv_pol_chain.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv_with_pol_chain.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace integrator_vv_pol_chain_kernels {

__global__ void integrate(PVviewWithPolChainVector view, const real4 *oldPositions, const real dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.size) return;

    const real4 pos = oldPositions[i];
    const real4 vel = readNoCache(view.velocities + i);
    const Real3_int frc(view.forces[i]);

    Particle p(pos, vel);

    p.u += (view.invMass * dt) * frc.v;
    p.r += dt * p.u;

    view.Q[i] += dt * view.dQdt[i];

    writeNoCache(view.positions  + i, p.r2Real4());
    writeNoCache(view.velocities + i, p.u2Real4());
}


} // namespace integrator_vv_pol_chain_kernels

IntegratorVVPolChain::IntegratorVVPolChain(const MirState *state, const std::string& name)
    : Integrator(state, name)
{}

void IntegratorVVPolChain::execute(ParticleVector *pv, cudaStream_t stream)
{
    const real dt = getState()->getDt();

    constexpr int nthreads = 128;

    PinnedBuffer<real4> *oldPositions = pv->local()->dataPerParticle.getData<real4>(channel_names::oldPositions);
    std::swap(pv->local()->positions(), *oldPositions);

    PVviewWithPolChainVector view(pv, pv->local());

    SAFE_KERNEL_LAUNCH(
        integrator_vv_pol_chain_kernels::integrate,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, oldPositions->devPtr(), dt);

    invalidatePV_(pv);
}

} // namespace mirheo
