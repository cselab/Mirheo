#pragma once

#include "lj.h"

class InteractionLJWithStress : public InteractionLJ
{
public:
    InteractionLJWithStress(const MirState *state, std::string name,
                            float rc, float epsilon, float sigma, float maxForce,
                            AwareMode awareness, int minSegmentsDist, float stressPeriod);

    ~InteractionLJWithStress();

    void setSpecificPair(ParticleVector *pv1, ParticleVector *pv2, 
                         float epsilon, float sigma, float maxForce) override;

protected:
    float stressPeriod;
};

