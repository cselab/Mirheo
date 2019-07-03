#pragma once

#include "generic_packer.h"

#include <vector>

class LocalParticleVector;

struct ParticlePackerHandler
{
    GenericPackerHandler particles;
};

class ParticlePacker
{
public:
    ParticlePacker();
    ParticlePacker(const std::vector<size_t>& extraTypeSize);
    ~ParticlePacker();
    
    void update(LocalParticleVector *lpv, PackPredicate& predicate, cudaStream_t stream);
    ParticlePackerHandler handler();
    virtual size_t getSizeBytes(int numElements) const;

protected:
    GenericPacker particleData;

    // optional elements which will be manually packed
    // but used in getSizeBytes
    const std::vector<size_t> extraTypeSize;
};
