// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "pv.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

PVview::PVview(ParticleVector *pv, LocalParticleVector *lpv)
{
    size = lpv->size();
    positions  = lpv->positions() .devPtr();
    velocities = lpv->velocities().devPtr();
    forces     = reinterpret_cast<real4*>(lpv->forces().devPtr());

    mass = pv->getMassPerParticle();
    invMass = 1.0_r / mass;
}

PVviewWithOldParticles::PVviewWithOldParticles(ParticleVector *pv, LocalParticleVector *lpv) :
    PVview(pv, lpv)
{
    if (lpv != nullptr)
        oldPositions = lpv->dataPerParticle.getData<real4>(channel_names::oldPositions)->devPtr();
}

PVviewWithDensities::PVviewWithDensities(ParticleVector *pv, LocalParticleVector *lpv) :
    PVview(pv, lpv)
{
    densities = lpv->dataPerParticle.getData<real>(channel_names::densities)->devPtr();
}

} // namespace mirheo
