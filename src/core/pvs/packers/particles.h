#pragma once

#include "generic_packer.h"

class LocalParticleVector;

struct ParticlePackerHandler
{
    GenericPackerHandler particles;
};

class ParticlePacker
{
public:
    void update(LocalParticleVector *lpv, PackPredicate& predicate, cudaStream_t stream);
    ParticlePackerHandler handler();
    virtual size_t getSizeBytes(int numElements) const;
    
protected:
    GenericPacker particleData;
};
