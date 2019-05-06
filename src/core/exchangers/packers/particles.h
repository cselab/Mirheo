#pragma once

#include "interface.h"

class ParticlePacker : public Packer
{
public:
    ParticlePacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate);

    size_t getPackedSizeBytes(int n) override;

    void packToBuffer(const DeviceBuffer<MapEntry>& map, const PinnedBuffer<int>& sizes, const PinnedBuffer<int>& offsets,
                      PinnedBuffer<size_t>& offsetsBytes, char *buffer, cudaStream_t stream);

    void unpackFromBuffer(PinnedBuffer<size_t>& offsetsBytes,
                          const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                          const char *buffer, cudaStream_t stream);
};
