#pragma once

#include "interface.h"

/**
 * Make constant particle velocities, regardless force
 * Coordinate is computed by Velocity-Verlet scheme (same as
 * Euler in this case)
 */
class IntegratorTranslate : public Integrator
{
public:

    IntegratorTranslate(const YmrState *state, std::string name, float3 vel);
    ~IntegratorTranslate();

    void stage1(ParticleVector *pv, cudaStream_t stream) override {};
    void stage2(ParticleVector *pv, cudaStream_t stream) override;

  private:

    float3 vel;   ///< Velocity
};
