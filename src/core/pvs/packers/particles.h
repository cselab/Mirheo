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
    ParticlePacker(PackPredicate predicate);
    ParticlePacker(PackPredicate predicate,
                   const std::vector<size_t>& extraTypeSize);
    ~ParticlePacker();
    
    void update(LocalParticleVector *lpv, cudaStream_t stream);
    ParticlePackerHandler handler();
    virtual size_t getSizeBytes(int numElements) const;

protected:
    PackPredicate predicate;
    GenericPacker particleData;

    // optional elements which will be manually packed
    // but used in getSizeBytes
    const std::vector<size_t> extraTypeSize;
};
