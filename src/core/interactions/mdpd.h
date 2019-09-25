#pragma once

#include "interface.h"
#include <memory>
#include <limits>

class InteractionMDPD : public Interaction
{
public:
    constexpr static float Default = std::numeric_limits<float>::infinity();

    InteractionMDPD(const MirState *state, std::string name, float rc, float rd, float a, float b, float gamma, float kbt, float power);

    ~InteractionMDPD();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    std::vector<InteractionChannel> getInputChannels() const override;
    std::vector<InteractionChannel> getOutputChannels() const override;
    
    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

    virtual void setSpecificPair(ParticleVector *pv1, ParticleVector *pv2, 
                                 float a=Default, float b=Default, float gamma=Default,
                                 float kbt=Default, float power=Default);
        
protected:

    InteractionMDPD(const MirState *state, std::string name, float rc, float rd, float a, float b, float gamma, float kbt, float power, bool allocateImpl);
    
    // Default values
    float rd, a, b, gamma, kbt, power;
};

