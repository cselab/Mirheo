#pragma once

#include "interface.h"

class ParticlePacker : public Packer
{
public:
    ParticlePacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate);

    size_t getPackedSizeBytes(int n) override;
    void packToBuffer(DeviceBuffer<MapEntry>& map, PinnedBuffer<size_t>& offsets, PinnedBuffer<int>& sizes, char *buufer, cudaStream_t stream);

protected:
    DeviceBuffer<size_t> localOffsets;
};
