#pragma once

#include "interface.h"
#include <memory>
#include <limits>
#include <core/utils/pytypes.h>

/**
 * Implementation of Velocity-Verlet integration in one step
 */
class InteractionDPD : Interaction
{
public:
    constexpr static float Default = std::numeric_limits<float>::infinity();

    std::unique_ptr<Interaction> impl;

    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) override;
    void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    
    void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
        float a=Default, float gamma=Default, float kbt=Default, float dt=Default, float power=Default);
    
    InteractionDPD(std::string name, float rc, float a, float gamma, float kbt, float dt, float power);

    ~InteractionDPD();
    
private:
    
    // Default values
    float a, gamma, kbt, dt, power;
};

