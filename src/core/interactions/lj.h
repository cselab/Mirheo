#pragma once

#include "interface.h"
#include <memory>

struct InteractionLJ : public Interaction
{        
    InteractionLJ(const YmrState *state, std::string name, float rc, float epsilon, float sigma, float maxForce, bool objectAware);

    ~InteractionLJ();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    void initStep(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    virtual void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                                 float epsilon, float sigma, float maxForce);

protected:
    InteractionLJ(const YmrState *state, std::string name, float rc, float epsilon, float sigma, float maxForce, bool objectAware, bool allocate);
    
    std::unique_ptr<Interaction> impl;
    bool objectAware;
};

