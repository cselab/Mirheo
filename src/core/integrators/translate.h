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

    IntegratorTranslate(std::string name, const YmrState *state, float3 vel);
    ~IntegratorTranslate();

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override {};
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

private:

    float3 vel;   ///< Velocity
};
