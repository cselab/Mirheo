#pragma once

#include "membrane.h"
#include "membrane/parameters.h"

#include <memory>

class InteractionMembraneWLCJuelicher : public InteractionMembrane
{
public:
    InteractionMembraneWLCJuelicher(const YmrState *state, std::string name,
                                    MembraneParameters parameters, JuelicherBendingParameters juelicherParams,
                                    bool stressFree, float growUntil);
    ~InteractionMembraneWLCJuelicher();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
protected:

    void precomputeQuantities(ParticleVector *pv1, cudaStream_t stream) override;
};
