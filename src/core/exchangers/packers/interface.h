#pragma once

#include "map.h"

#include <core/pvs/data_manager.h>
#include <core/ymero_state.h>

#include <vector_types.h>
#include <cuda_runtime.h>

using PackPredicate = std::function< bool (const DataManager::NamedChannelDesc&) >;

class ParticleVector;
class LocalParticleVector;

class Packer
{
public:
    Packer(const YmrState *state, ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate);
    
    virtual size_t getPackedSizeBytes(int n) = 0;

    template <typename TPadding = float4>
    __HD__ constexpr static size_t getPackedSize(size_t datumSize, int n)
    {
        size_t size = n * datumSize;
        size_t npads = (size + sizeof(TPadding)-1) / sizeof(TPadding);
        return npads * sizeof(TPadding);
    }

    template <typename T, typename TPadding = float4>
    __HD__ static size_t getPackedSize(int n)
    {
        return getPackedSize<TPadding>(sizeof(T), n);
    }


protected:
    size_t _getPackedSizeBytes(DataManager& manager, int n);

protected:
    const YmrState *state;
    PackPredicate predicate;
    ParticleVector *pv;
    LocalParticleVector *lpv;
};
