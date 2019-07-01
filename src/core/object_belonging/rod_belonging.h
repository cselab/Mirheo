#pragma once

#include "object_belonging.h"

class RodBelongingChecker : public ObjectBelongingChecker_Common
{
public:
    RodBelongingChecker(const MirState *state, std::string name, float radius);
    
    void tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;

private:
    float radius;
};
