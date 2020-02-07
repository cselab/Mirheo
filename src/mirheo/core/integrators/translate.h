#pragma once

#include "interface.h"

#include <mirheo/core/utils/macros.h>

namespace mirheo
{

/**
 * Make constant particle velocities, regardless force
 * Coordinate is computed by Velocity-Verlet scheme (same as
 * Euler in this case)
 */
class IntegratorTranslate : public Integrator
{
public:

    IntegratorTranslate(const MirState *state, const std::string& name, real3 vel);
    ~IntegratorTranslate();

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:

    real3 vel_;   ///< Velocity
};

} // namespace mirheo
