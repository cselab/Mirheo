// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "const_omega.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo {

IntegratorConstOmega::IntegratorConstOmega(const MirState *state, const std::string& name, real3 center, real3 omega) :
    Integrator(state, name),
    center_(center),
    omega_(omega)
{}

IntegratorConstOmega::~IntegratorConstOmega() = default;

void IntegratorConstOmega::execute(ParticleVector *pv, cudaStream_t stream)
{
    const auto domain = getState()->domain;
    const auto center = center_;
    const auto omega = omega_;

    auto rotate = [domain, center, omega] __device__ (Particle& p, real3 f, real invm, real dt)
    {
        constexpr real tolerance = 1e-6;
        const real3 gr = domain.local2global(p.r);
        real3 gr_c = gr - center;
        p.u = cross(omega, gr_c);
        const real IrI = length(gr_c);

        if (IrI < tolerance)
            return;

        gr_c += p.u * dt;
        gr_c = normalize(gr_c) * IrI;
        p.r  = domain.global2local(gr_c + center);
    };

    integrate(pv, getState()->getDt(), rotate, stream);
    invalidatePV_(pv);
}

} // namespace mirheo
