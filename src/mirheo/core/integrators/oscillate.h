#pragma once

#include "interface.h"

#include <mirheo/core/utils/macros.h>

/**
 * Apply periodic sine wave to the particle velocities.
 * Coordinate is computed by Velocity-Verlet scheme (same as
 * Euler in this case)
 */
class IntegratorOscillate : public Integrator
{
public:

    IntegratorOscillate(const MirState *state, std::string name, real3 vel, real period);

    ~IntegratorOscillate();

    void stage1(__UNUSED ParticleVector *pv, __UNUSED cudaStream_t stream) override {};
    void stage2(ParticleVector *pv, cudaStream_t stream) override;

private:

    real3 vel;    ///< Velocity amplitude
    real period;  ///< Sine wave period
};
