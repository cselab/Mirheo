#include "interface.h"

#include <core/pvs/particle_vector.h>

Packer::Packer(const YmrState *state, ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate) :
    state(state),
    pv(pv),
    lpv(lpv),
    predicate(predicate)
{}

size_t Packer::_getPackedSizeBytes(DataManager& manager, int n)
{
    size_t size = 0;
    
    for (const auto& name_desc : manager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        
        auto& name = name_desc.first;
        auto& desc = name_desc.second;

        mpark::visit([&](auto pinnedBuffPtr) {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;
            size += getPackedSize<T>(n);
        }, desc->varDataPtr);
    }

    return size;
}
