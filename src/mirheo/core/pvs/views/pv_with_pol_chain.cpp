// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "pv_with_pol_chain.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo {

PVviewWithPolChainVector::PVviewWithPolChainVector(ParticleVector *pv, LocalParticleVector *lpv) :
    PVview(pv, lpv)
{
    this->Q    = lpv->dataPerParticle.getData<real3>(channel_names::polChainVectors)->devPtr();
    this->dQdt = lpv->dataPerParticle.getData<real3>(channel_names::derChainVectors)->devPtr();
}

} // namespace mirheo
