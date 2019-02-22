#pragma once

#include "interface.h"
#include <memory>

class InteractionMembrane : public Interaction
{
public:

    InteractionMembrane(const YmrState *state, std::string name);
    ~InteractionMembrane();
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) final;

protected:

    virtual void precomputeQuantities(ParticleVector *pv1, cudaStream_t stream);
    
    std::unique_ptr<Interaction> impl; // concrete implementation of forces
};
