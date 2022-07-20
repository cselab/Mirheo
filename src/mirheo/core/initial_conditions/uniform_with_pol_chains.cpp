// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "uniform_with_pol_chains.h"
#include "helpers.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo {

UniformWithPolChainIC::UniformWithPolChainIC(real numDensity) :
    numDensity_(numDensity)
{}

void UniformWithPolChainIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto filterInKeepAll = [](real3) {return true;};
    setUniformParticles(numDensity_, comm, pv, filterInKeepAll, stream);

    pv->requireDataPerParticle<real3>(channel_names::polChainVectors, DataManager::PersistenceMode::Active);

    PinnedBuffer<real3>& Qs = *pv->local()->dataPerParticle.getData<real3>(channel_names::polChainVectors);

    for (auto& Q : Qs)
    {
        Q.x = Q.y = Q.z = 0.0_r;
    }

    Qs.uploadToDevice(stream);
}

} // namespace mirheo
