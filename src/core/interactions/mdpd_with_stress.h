#pragma once

#include "mdpd.h"

class InteractionMDPDWithStress : public InteractionMDPD
{
public:
    InteractionMDPDWithStress(const MirState *state, std::string name,
                              float rc, float rd, float a, float b, float gamma, float kbt, float power,
                              float stressPeriod);

    ~InteractionMDPDWithStress();    
    
    void setSpecificPair(ParticleVector *pv1, ParticleVector *pv2, 
                         float a=Default, float b=Default, float gamma=Default,
                         float kbt=Default, float power=Default) override;

protected:
    float stressPeriod;
};

