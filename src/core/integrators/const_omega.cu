#include "const_omega.h"
#include "integration_kernel.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>

IntegratorConstOmega::IntegratorConstOmega(const YmrState *state, std::string name, float3 center, float3 omega) :
    Integrator(state, name),
    center(center), omega(omega)
{}

IntegratorConstOmega::~IntegratorConstOmega() = default;

void IntegratorConstOmega::stage1(ParticleVector *pv, cudaStream_t stream)
{}

void IntegratorConstOmega::stage2(ParticleVector *pv, cudaStream_t stream)
{
    const auto domain = state->domain;
    const auto _center = center;
    const auto _omega = omega;

    auto rotate = [domain, _center, _omega] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
        constexpr float tolerance = 1e-6;
        float3 gr = domain.local2global(p.r);
        float3 gr_c = gr - _center;
        p.u = cross(_omega, gr_c);
        float IrI = length(gr_c);

        if (IrI < tolerance)
            return;

        gr_c += p.u*dt;
        gr_c = normalize(gr_c) * IrI;
        p.r  = domain.global2local(gr_c + _center);
    };

    integrate(pv, state->dt, rotate, stream);
    invalidatePV(pv);
}
