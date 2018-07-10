#pragma once

#include "interface.h"
#include <memory>

/**
 * Implementation of Velocity-Verlet integration in one step
 */
struct IntegratorVV_noforce : Integrator
{
    std::unique_ptr<Integrator> impl;

    void stage1(ParticleVector* pv, float t, cudaStream_t stream) override;
    void stage2(ParticleVector* pv, float t, cudaStream_t stream) override;

    IntegratorVV_noforce(std::string name, float dt, std::tuple<float, float, float> extra_force);

    ~IntegratorVV_noforce();
};

