#pragma once

#include "interface.h"

class ExchangeHelper;

class ParticlesPacker : public Packer
{
public:
    ParticlesPacker(ParticleVector *pv, PackPredicate predicate);

    size_t getPackedSizeBytes(int n) const override;

    void packToBuffer(const LocalParticleVector *lpv, ExchangeHelper *helper, cudaStream_t stream);

    void unpackFromBuffer(LocalParticleVector *lpv,
                          const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                          const char *buffer, int oldSize, cudaStream_t stream);
};
