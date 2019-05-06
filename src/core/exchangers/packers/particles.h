#pragma once

#include "interface.h"

class ParticlesPacker : public Packer
{
public:
    ParticlesPacker(const YmrState *state, ParticleVector *pv, PackPredicate predicate);

    size_t getPackedSizeBytes(int n) const override;

    void packToBuffer(const LocalParticleVector *lpv,
                      const DeviceBuffer<MapEntry>& map, const PinnedBuffer<int>& sizes,
                      const PinnedBuffer<int>& offsets, char *buffer, cudaStream_t stream);

    void unpackFromBuffer(LocalParticleVector *lpv,
                          const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                          const char *buffer, int oldSize, cudaStream_t stream);
};
