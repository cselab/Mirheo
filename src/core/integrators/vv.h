#pragma once

#include "interface.h"

/**
 * Implementation of Velocity-Verlet integration in one step
 */
template<class ForcingTerm>
struct IntegratorVV : Integrator
{
    ForcingTerm forcingTerm;

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

    IntegratorVV(std::string name, float dt, ForcingTerm forcingTerm) :
        Integrator(name, dt), forcingTerm(forcingTerm)
    {}

    ~IntegratorVV();
};
