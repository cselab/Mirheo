#pragma once

#include "interface.h"

namespace mirheo
{

/**
 * Implementation of Velocity-Verlet integration in one step
 */
template<class ForcingTerm>
struct IntegratorVV : Integrator
{
    ForcingTerm forcingTerm;

    IntegratorVV(const MirState *state, std::string name, ForcingTerm forcingTerm);
    ~IntegratorVV();
    Config getConfig() const override;

    void stage1(ParticleVector *pv, cudaStream_t stream) override;
    void stage2(ParticleVector *pv, cudaStream_t stream) override;
};

} // namespace mirheo
