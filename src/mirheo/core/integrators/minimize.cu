// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "minimize.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

IntegratorMinimize::IntegratorMinimize(const MirState *state, const std::string& name, real maxDisplacement) :
    Integrator(state, name), maxDisplacement_{maxDisplacement}
{}

void IntegratorMinimize::execute(ParticleVector *pv, cudaStream_t stream)
{
    const auto t  = static_cast<real>(getState()->currentTime);
    const auto dt = static_cast<real>(getState()->getDt());

    auto st2 = [max = maxDisplacement_] __device__ (Particle& p, real3 f, real invm, real dt)
    {
        // Limit the displacement magnitude to `max`.
        real3 dr = dt * dt * invm * f;
        real dr2 = dot(dr, dr);
        if (dr2 > max * max)
            dr *= max * math::rsqrt(dr2);
        p.r += dr;
    };

    integrate(pv, dt, st2, stream);
    invalidatePV_(pv);
}

} // namespace mirheo
