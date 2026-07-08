// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "uniform_with_pol_chains.h"
#include "helpers.h"

#include <mirheo/core/pvs/particle_vector.h>

#include <random>

namespace mirheo {

UniformWithPolChainIC::UniformWithPolChainIC(real numDensity, real q0) :
    numDensity_(numDensity),
    q0_(q0)
{}

void UniformWithPolChainIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto filterInKeepAll = [](real3) {return true;};
    setUniformParticles(numDensity_, comm, pv, filterInKeepAll, stream);

    pv->requireDataPerParticle<real3>(channel_names::polChainVectors, DataManager::PersistenceMode::Active);

    PinnedBuffer<real3>& Qs = *pv->local()->dataPerParticle.getData<real3>(channel_names::polChainVectors);

    int rank = 0;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    const long seed = 1097 + 1265 * rank;
    std::normal_distribution<real> qdistr(0.0_r, q0_);
    std::mt19937 gen(seed);

    for (auto& Q : Qs)
    {
        Q.x = qdistr(gen);
        Q.y = qdistr(gen);
        Q.z = qdistr(gen);
    }

    Qs.uploadToDevice(stream);
}

} // namespace mirheo
