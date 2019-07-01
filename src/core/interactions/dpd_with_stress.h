#pragma once

#include "dpd.h"

class InteractionDPDWithStress : public InteractionDPD
{
public:
    InteractionDPDWithStress(const MirState *state, std::string name,
                             float rc, float a, float gamma, float kbt, float power, float stressPeriod);

    ~InteractionDPDWithStress();    
    
    void setSpecificPair(ParticleVector *pv1, ParticleVector *pv2, 
                         float a   = Default, float gamma = Default,
                         float kbt = Default, float power = Default) override;
};

