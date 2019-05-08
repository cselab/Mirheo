#pragma once

#include "interface.h"

#include <string>
#include <vector>

class ExchangeHelper;
struct BufferInfos;

class ParticlesPacker : public Packer
{
public:
    ParticlesPacker(ParticleVector *pv, PackPredicate predicate);

    size_t getPackedSizeBytes(int n) const override;

    void packToBuffer(const LocalParticleVector *lpv, const DeviceBuffer<MapEntry>& map, BufferInfos *helper,
                      const std::vector<std::string>& alreadyPacked, cudaStream_t stream);

    void unpackFromBuffer(LocalParticleVector *lpv, const BufferInfos *helper, int oldSize, cudaStream_t stream);

private:
    template <typename Visitor>
    void _applyToChannels(const LocalParticleVector *lpv, Visitor &&visitor);
};
