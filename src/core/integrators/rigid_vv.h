#pragma once

#include "interface.h"

/**
 * Integrate motion of the rigid bodies.
 */
class IntegratorVVRigid : public Integrator
{
public:
    IntegratorVVRigid(const YmrState *state, std::string name);

    ~IntegratorVVRigid();

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

    void setPrerequisites(ParticleVector* pv) override;
};
