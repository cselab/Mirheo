#pragma once

#include "interface.h"

#include <mirheo/core/utils/macros.h>

namespace mirheo
{

/**
 * Apply periodic sine wave to the particle velocities.
 * Coordinate is computed by Velocity-Verlet scheme (same as
 * Euler in this case)
 */
class IntegratorOscillate : public Integrator
{
public:

    IntegratorOscillate(const MirState *state, const std::string& name, real3 vel, real period);

    ~IntegratorOscillate();

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:

    real3 vel_;    ///< Velocity amplitude
    real period_;  ///< Sine wave period
};

} // namespace mirheo
