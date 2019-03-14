#pragma once

#include "interface.h"
#include <memory>

class BasicInteractionDensity : public Interaction
{
public:
    BasicInteractionDensity(const YmrState *state, std::string name, float rc);
    ~BasicInteractionDensity();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    
    std::vector<InteractionChannel> getIntermediateOutputChannels() const override;
    std::vector<InteractionChannel> getFinalOutputChannels() const override;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
        
protected:

    std::unique_ptr<Interaction> impl;
};


template <class DensityKernel>
class InteractionDensity : public BasicInteractionDensity
{
public:
    InteractionDensity(const YmrState *state, std::string name, float rc, DensityKernel densityKernel);
    ~InteractionDensity();
        
protected:

    DensityKernel densityKernel;
};
