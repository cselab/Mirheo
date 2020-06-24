// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "translate.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{


/**
 * @param vel Move with this velocity
 */
IntegratorTranslate::IntegratorTranslate(const MirState *state, const std::string& name, real3 vel) :
    Integrator(state, name),
    vel_(vel)
{}

IntegratorTranslate::~IntegratorTranslate() = default;

void IntegratorTranslate::execute(ParticleVector *pv, cudaStream_t stream)
{
    const auto vel = vel_;

    auto translate = [vel] __device__ (Particle& p, real3 f, real invm, real dt)
    {
        p.u = vel;
        p.r += p.u * dt;
    };

    integrate(pv, getState()->dt, translate, stream);
    invalidatePV_(pv);
}

} // namespace mirheo
