#pragma once

#include "object_belonging.h"

namespace mirheo
{

class RodBelongingChecker : public ObjectBelongingChecker_Common
{
public:
    RodBelongingChecker(const MirState *state, std::string name, real radius);
    
    void tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;

private:
    real radius;
};

} // namespace mirheo
