#pragma once

#include "membrane.new.h"
#include "membrane/parameters.h"

#include <memory>

class MembraneWLCJuelicher : public InteractionMembraneNew
{
public:
    MembraneWLCJuelicher(const YmrState *state, std::string name,
                         MembraneParameters parameters, JuelicherBendingParameters juelicherParams,
                         bool stressFree, float growUntil);
    ~MembraneWLCJuelicher();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
protected:

    void precomputeQuantities(ParticleVector *pv1, cudaStream_t stream) override;
};
