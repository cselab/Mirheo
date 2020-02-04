#include "particles.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

ParticlePacker::ParticlePacker(PackPredicate predicate) :
    predicate_(predicate)
{}

ParticlePacker::~ParticlePacker() = default;

void ParticlePacker::update(LocalParticleVector *lpv, cudaStream_t stream)
{
    particleData_.updateChannels(lpv->dataPerParticle, predicate_, stream);
}

ParticlePackerHandler ParticlePacker::handler()
{
    return {particleData_.handler()};
}

size_t ParticlePacker::getSizeBytes(int numElements) const
{
    return particleData_.getSizeBytes(numElements);
}

} // namespace mirheo
