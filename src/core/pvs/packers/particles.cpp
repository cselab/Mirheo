#include "particles.h"

#include <core/pvs/particle_vector.h>

ParticlePacker::ParticlePacker() = default;

ParticlePacker::ParticlePacker(const std::vector<size_t>& extraTypeSize) :
    extraTypeSize(extraTypeSize)
{}

ParticlePacker::~ParticlePacker() = default;

void ParticlePacker::update(LocalParticleVector *lpv, PackPredicate& predicate, cudaStream_t stream)
{
    particleData.updateChannels(lpv->dataPerParticle, predicate, stream);
}

ParticlePackerHandler ParticlePacker::handler()
{
    return {particleData.handler()};
}

size_t ParticlePacker::getSizeBytes(int numElements) const
{
    size_t totSize = particleData.getSizeBytes(numElements);

    for (const auto& datumSize : extraTypeSize)
        totSize += getPaddedSize(datumSize, numElements);
    
    return totSize;
}
