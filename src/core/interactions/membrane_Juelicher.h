#pragma once

#include "membrane.h"

#include <memory>

class InteractionMembraneJuelicher : public InteractionMembrane
{
public:
    InteractionMembraneJuelicher(const YmrState *state, std::string name);
    
    ~InteractionMembraneJuelicher();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
protected:

    void precomputeQuantities(ParticleVector *pv1, cudaStream_t stream) override;
};
