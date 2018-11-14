#pragma once

#include "interface.h"
#include <memory>

struct InteractionLJ : Interaction
{
    std::unique_ptr<Interaction> impl;

    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) override;
    void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    
    void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
        float epsilon, float sigma, float maxForce);
    
    InteractionLJ(std::string name, float rc, float epsilon, float sigma, float maxForce, bool objectAware);

    ~InteractionLJ();
    
private:
    bool objectAware;
};

