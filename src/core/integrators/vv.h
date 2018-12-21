#pragma once

#include "interface.h"

/**
 * Implementation of Velocity-Verlet integration in one step
 */
template<class ForcingTerm>
struct IntegratorVV : Integrator
{
    ForcingTerm forcingTerm;

    IntegratorVV(const YmrState *state, std::string name, ForcingTerm forcingTerm);
    ~IntegratorVV();

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;
};
