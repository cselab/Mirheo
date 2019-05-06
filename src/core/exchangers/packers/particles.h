#pragma once

#include "interface.h"

class ParticlePacker : public Packer
{
public:
    ParticlePacker(const YmrState *state, ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate);

    size_t getPackedSizeBytes(int n) override;

    void packToBuffer(const DeviceBuffer<MapEntry>& map, const PinnedBuffer<int>& sizes, const PinnedBuffer<int>& offsets,
                      char *buffer, cudaStream_t stream);

    void unpackFromBuffer(const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                          const char *buffer, int oldSize, cudaStream_t stream);

protected:
    DeviceBuffer<size_t> offsetsBytes;
};
