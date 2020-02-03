#pragma once

#include "interface.h"

namespace mirheo
{

/**
 * Implementation of Velocity-Verlet integration in one step
 */
template<class ForcingTerm>
class IntegratorVV : public Integrator
{
public:
    IntegratorVV(const MirState *state, const std::string& name, ForcingTerm forcingTerm);
    ~IntegratorVV();

    void stage1(ParticleVector *pv, cudaStream_t stream) override;
    void stage2(ParticleVector *pv, cudaStream_t stream) override;

private:
    ForcingTerm forcingTerm_;
};

} // namespace mirheo
