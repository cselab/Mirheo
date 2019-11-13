#pragma once

#include "pv.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

template <typename BasicView> 
struct PVviewWithStresses : public BasicView
{
    PVviewWithStresses(ParticleVector *pv, LocalParticleVector *lpv) :
        BasicView(pv, lpv)
    {
        stresses = lpv->dataPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();            
    }

    Stress *stresses {nullptr};
};

} // namespace mirheo
