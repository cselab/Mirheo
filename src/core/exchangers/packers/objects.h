#pragma once

#include "interface.h"

class ObjectVector;
class LocalObjectVector;

class ObjectPacker : public Packer
{
public:
    ObjectPacker(const YmrState *state, ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate);

    size_t getPackedSizeBytes(int n) const override;

    void packToBuffer(const DeviceBuffer<MapEntry>& map, const PinnedBuffer<int>& sizes,
                      const PinnedBuffer<int>& offsets, char *buffer, cudaStream_t stream);
    
    void unpackFromBuffer(const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                          const char *buffer, int oldObjSize, cudaStream_t stream);

protected:
    ObjectVector *ov;
    LocalObjectVector * lov;
    DeviceBuffer<size_t> offsetsBytes;
};
