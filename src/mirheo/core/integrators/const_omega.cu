#include "const_omega.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

IntegratorConstOmega::IntegratorConstOmega(const MirState *state, std::string name, real3 center, real3 omega) :
    Integrator(state, name),
    center(center), omega(omega)
{}

IntegratorConstOmega::~IntegratorConstOmega() = default;

void IntegratorConstOmega::stage1(__UNUSED ParticleVector *pv, __UNUSED cudaStream_t stream)
{}

void IntegratorConstOmega::stage2(ParticleVector *pv, cudaStream_t stream)
{
    const auto domain = state->domain;
    const auto _center = center;
    const auto _omega = omega;

    auto rotate = [domain, _center, _omega] __device__ (Particle& p, const real3 f, const real invm, const real dt) {
        constexpr real tolerance = 1e-6;
        real3 gr = domain.local2global(p.r);
        real3 gr_c = gr - _center;
        p.u = cross(_omega, gr_c);
        real IrI = length(gr_c);

        if (IrI < tolerance)
            return;

        gr_c += p.u*dt;
        gr_c = normalize(gr_c) * IrI;
        p.r  = domain.global2local(gr_c + _center);
    };

    integrate(pv, state->dt, rotate, stream);
    invalidatePV(pv);
}
