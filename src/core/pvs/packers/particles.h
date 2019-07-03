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

    template <typename T>
    static constexpr size_t getSizeBytesExtraEntry(int numElements)
    {
        return getPaddedSize<T>(numElements);
    }
    
protected:
    GenericPacker particleData;
};
