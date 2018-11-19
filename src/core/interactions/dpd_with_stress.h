#pragma once

#include "dpd.h"

class InteractionDPDWithStress : public InteractionDPD
{
public:
    InteractionDPDWithStress(std::string name, std::string stressName, float rc, float a, float gamma, float kbt, float dt, float power, float stressPeriod);

    ~InteractionDPDWithStress();

    void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                         float a=Default, float gamma=Default, float kbt=Default,
                         float dt=Default, float power=Default) override;

protected:
    float stressPeriod;
};

