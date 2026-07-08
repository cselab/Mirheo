// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "shear.h"
#include "integration_kernel.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo {

IntegratorShear::IntegratorShear(const MirState *state, const std::string& name,
                                 std::array<real,9> shear, real3 origin) :
    Integrator(state, name),
    shear_(shear),
    origin_(origin)
{}

IntegratorShear::~IntegratorShear() = default;

void IntegratorShear::execute(ParticleVector *pv, cudaStream_t stream)
{
    const auto domain = getState()->domain;
    const auto origin = origin_;

    const real3 shearx {shear_[0], shear_[1], shear_[2]};
    const real3 sheary {shear_[3], shear_[4], shear_[5]};
    const real3 shearz {shear_[6], shear_[7], shear_[8]};

    auto applyShear = [domain, origin,
                       shearx, sheary, shearz]
        __device__ (Particle& p, real3 f, real invm, real dt)
    {
        const real3 r = domain.local2global(p.r) - origin;
        const real3 vel {dot(shearx, r),
                         dot(sheary, r),
                         dot(shearz, r)};
        p.u = vel;
        p.r += vel * dt;
    };

    integrate(pv, getState()->getDt(), applyShear, stream);
    invalidatePV_(pv);
}

} // namespace mirheo
