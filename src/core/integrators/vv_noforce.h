#pragma once

#include "interface.h"
#include <memory>
#include <core/utils/pytypes.h>

/**
 * Implementation of Velocity-Verlet integration in one step
 */
struct IntegratorVV_noforce : Integrator
{
    std::unique_ptr<Integrator> impl;

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

    IntegratorVV_noforce(std::string name, float dt);

    ~IntegratorVV_noforce();
};

