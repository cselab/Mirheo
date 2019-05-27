#pragma once

#include "interface.h"
#include "rod/parameters.h"

class InteractionRod : public Interaction
{
public:
    InteractionRod(const YmrState *state, std::string name, RodParameters params, bool dumpStates, bool dumpEnergies);
    virtual ~InteractionRod();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
    void local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;
    void halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;

    bool isSelfObjectInteraction() const override;
};
