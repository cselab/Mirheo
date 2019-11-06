#pragma once

#include "interface.h"

namespace mirheo
{

/**
 * Integrate motion of the rigid bodies.
 */
class IntegratorVVRigid : public Integrator
{
public:
    IntegratorVVRigid(const MirState *state, std::string name);

    ~IntegratorVVRigid();

    void setPrerequisites(ParticleVector *pv) override;

    void stage1(ParticleVector *pv, cudaStream_t stream) override;
    void stage2(ParticleVector *pv, cudaStream_t stream) override;
};

} // namespace mirheo
