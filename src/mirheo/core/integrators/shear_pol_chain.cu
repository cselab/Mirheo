// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "shear_pol_chain.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv_with_pol_chain.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace integrator_shear_pol_chain_kernels {

__global__ void integrate(PVviewWithPolChainVector view, const real4 *oldPositions, const real dt,
                          const real3 shearx, const real3 sheary, const real3 shearz,
                          const DomainInfo domain, const real3 origin)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.size) return;

    const real4 pos = oldPositions[i];
    const real4 vel = readNoCache(view.velocities + i);
    const Real3_int frc(view.forces[i]);

    Particle p(pos, vel);

    const real3 r = domain.local2global(p.r) - origin;
    const real3 v {dot(shearx, r),
                   dot(sheary, r),
                   dot(shearz, r)};

    p.u = v;
    p.r += dt * v;

    view.Q[i] += dt * view.dQdt[i];

    writeNoCache(view.positions  + i, p.r2Real4());
    writeNoCache(view.velocities + i, p.u2Real4());
}


} // namespace integrator_shear_pol_chain_kernels

IntegratorShearPolChain::IntegratorShearPolChain(const MirState *state, const std::string& name,
                                                 std::array<real,9> shear, real3 origin) :
    Integrator(state, name),
    shear_(shear),
    origin_(origin)
{}

void IntegratorShearPolChain::execute(ParticleVector *pv, cudaStream_t stream)
{
    const auto domain = getState()->domain;
    const auto origin = origin_;

    const real3 shearx {shear_[0], shear_[1], shear_[2]};
    const real3 sheary {shear_[3], shear_[4], shear_[5]};
    const real3 shearz {shear_[6], shear_[7], shear_[8]};

    const real dt = getState()->getDt();

    constexpr int nthreads = 128;

    PinnedBuffer<real4> *oldPositions = pv->local()->dataPerParticle.getData<real4>(channel_names::oldPositions);
    std::swap(pv->local()->positions(), *oldPositions);

    PVviewWithPolChainVector view(pv, pv->local());

    SAFE_KERNEL_LAUNCH(
        integrator_shear_pol_chain_kernels::integrate,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, oldPositions->devPtr(), dt,
        shearx, sheary, shearz, domain, origin);

    invalidatePV_(pv);
}

} // namespace mirheo
