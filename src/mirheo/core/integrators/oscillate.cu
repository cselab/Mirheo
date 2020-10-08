// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "oscillate.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

IntegratorOscillate::IntegratorOscillate(const MirState *state, const std::string& name, real3 vel, real period) :
    Integrator(state, name),
    vel_(vel),
    period_(period)
{
    if (period_ <= 0)
        die("Oscillating period should be strictly positive");
}

IntegratorOscillate::~IntegratorOscillate() = default;

/**
 * Oscillate with cos wave in time, regardless force
 */
void IntegratorOscillate::execute(ParticleVector *pv, cudaStream_t stream)
{
    const auto t = static_cast<real>(getState()->currentTime);

    const auto vel = vel_;
    constexpr auto twoPi = static_cast<real>(2.0 * M_PI);

    const real cosOmega = math::cos(twoPi * t / period_);

    auto oscillate = [vel, cosOmega] __device__ (Particle& p, real3 f, real invm, real dt)
    {
        p.u = vel * cosOmega;
        p.r += p.u * dt;
    };

    integrate(pv, getState()->getDt(), oscillate, stream);
    invalidatePV_(pv);
}

} // namespace mirheo
