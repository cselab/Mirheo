#include "oscillate.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

/**
 * @param vel Velocity magnitude
 * @param period Sine wave period
 */
IntegratorOscillate::IntegratorOscillate(const MirState *state, std::string name, real3 vel, real period) :
    Integrator(state, name),
    vel(vel), period(period)
{
    if (period <= 0)
        die("Oscillating period should be strictly positive");
}

IntegratorOscillate::~IntegratorOscillate() = default;

/**
 * Oscillate with cos wave in time, regardless force
 */
void IntegratorOscillate::stage2(ParticleVector *pv, cudaStream_t stream)
{
    real t = state->currentTime;
    
    const auto _vel = vel;
    real cosOmega = math::cos(2*M_PI * t / period);

    auto oscillate = [_vel, cosOmega] __device__ (Particle& p, const real3 f, const real invm, const real dt) {
        p.u = _vel * cosOmega;
        p.r += p.u*dt;
    };

    integrate(pv, state->dt, oscillate, stream);
    invalidatePV(pv);
}

} // namespace mirheo
