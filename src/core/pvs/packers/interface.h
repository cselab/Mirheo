#pragma once

#include "map.h"

#include <core/pvs/data_manager.h>

#include <vector_types.h>
#include <cuda_runtime.h>

using PackPredicate = std::function< bool (const DataManager::NamedChannelDesc&) >;

class ParticleVector;
class LocalParticleVector;

class Packer
{
public:
    Packer(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate);
    
    virtual size_t getPackedSizeBytes(int n) = 0;

    template <typename T, typename TPadding = float4>
    __HD__ static size_t getPackedSize(int n)
    {
        size_t size = n * sizeof(T);
        size_t npads = (size + sizeof(TPadding)-1) / sizeof(TPadding);
        return npads * sizeof(TPadding);
    }
    
protected:
    size_t _getPackedSizeBytes(DataManager& manager, int n);

protected:
    PackPredicate predicate;
    ParticleVector *pv;
    LocalParticleVector *lpv;
};
