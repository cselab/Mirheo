// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "pv_with_pol_chain_smooth_velocity.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo {

PVviewWithPolChainVectorAndSmoothVelocity::PVviewWithPolChainVectorAndSmoothVelocity(ParticleVector *pv, LocalParticleVector *lpv) :
    PVviewWithPolChainVector(pv, lpv)
{
    this->smoothVel = lpv->dataPerParticle.getData<real4>(channel_names::smoothVelocities)->devPtr();
}

} // namespace mirheo
