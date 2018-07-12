#pragma once

#include "interface.h"
#include <memory>
#include <core/utils/pytypes.h>

/**
 * Implementation of Velocity-Verlet integration in one step
 */
struct IntegratorVV_constDP : Integrator
{
    std::unique_ptr<Integrator> impl;

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

    IntegratorVV_constDP(std::string name, float dt, pyfloat3 extra_force);

    ~IntegratorVV_constDP();
};

