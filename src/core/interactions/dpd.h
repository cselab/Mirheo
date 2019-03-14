#pragma once

#include "interface.h"
#include <memory>
#include <limits>
#include <core/utils/pytypes.h>

class InteractionDPD : public Interaction
{
public:
    constexpr static float Default = std::numeric_limits<float>::infinity();

    InteractionDPD(const YmrState *state, std::string name, float rc, float a, float gamma, float kbt, float power);

    ~InteractionDPD();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;
    std::vector<InteractionChannel> getFinalOutputChannels() const override;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    virtual void setSpecificPair(ParticleVector *pv1, ParticleVector *pv2, 
                                 float a   = Default, float gamma = Default,
                                 float kbt = Default, float power = Default);
    
protected:

    InteractionDPD(const YmrState *state, std::string name, float rc, float a, float gamma, float kbt, float power, bool allocateImpl);
    
    std::unique_ptr<Interaction> impl;
    
    // Default values
    float a, gamma, kbt, power;
};

