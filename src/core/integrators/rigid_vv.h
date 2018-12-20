#pragma once

#include "interface.h"

/**
 * Integrate motion of the rigid bodies.
 */
class IntegratorVVRigid : public Integrator
{
public:
    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

    void setPrerequisites(ParticleVector* pv) override;

    IntegratorVVRigid(std::string name, const YmrState *state) :
        Integrator(name, state)
    {}

    ~IntegratorVVRigid() = default;
};
