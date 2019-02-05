#pragma once

#include "lj.h"

struct InteractionLJWithStress : public InteractionLJ
{        
    InteractionLJWithStress(const YmrState *state, std::string name,
                            float rc, float epsilon, float sigma, float maxForce, bool objectAware, float stressPeriod);

    ~InteractionLJWithStress();

    void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
                         float epsilon, float sigma, float maxForce) override;

protected:
    float stressPeriod;
};

