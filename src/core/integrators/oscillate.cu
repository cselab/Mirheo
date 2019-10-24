#include "oscillate.h"
#include "integration_kernel.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>

/**
 * @param vel Velocity magnitude
 * @param period Sine wave period
 */
IntegratorOscillate::IntegratorOscillate(const MirState *state, std::string name, float3 vel, float period) :
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
    float t = state->currentTime;
    
    const auto _vel = vel;
    float cosOmega = math::cos(2*M_PI * t / period);

    auto oscillate = [_vel, cosOmega] __device__ (Particle& p, const float3 f, const float invm, const float dt) {
        p.u = _vel * cosOmega;
        p.r += p.u*dt;
    };

    integrate(pv, state->dt, oscillate, stream);
    invalidatePV(pv);
}
