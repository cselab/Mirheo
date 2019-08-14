#pragma once

#include "interface.h"

#include <core/utils/macros.h>

/**
 * Apply periodic sine wave to the particle velocities.
 * Coordinate is computed by Velocity-Verlet scheme (same as
 * Euler in this case)
 */
class IntegratorOscillate : public Integrator
{
public:

    IntegratorOscillate(const MirState *state, std::string name, float3 vel, float period);

    ~IntegratorOscillate();

    void stage1(__UNUSED ParticleVector *pv, __UNUSED cudaStream_t stream) override {};
    void stage2(ParticleVector *pv, cudaStream_t stream) override;

private:

    float3 vel;    ///< Velocity amplitude
    float period;  ///< Sine wave period
};
