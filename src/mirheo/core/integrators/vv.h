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

    void execute(ParticleVector *pv, cudaStream_t stream) override;

private:
    ForcingTerm forcingTerm_;
};

} // namespace mirheo
