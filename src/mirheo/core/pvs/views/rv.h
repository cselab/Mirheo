#pragma once

#include "ov.h"

namespace mirheo
{

class RodVector;
class LocalRodVector;

struct RVview : public OVview
{
    RVview(RodVector *rv, LocalRodVector *lrv);
    
    int   nSegments {0};
    int   *states   {nullptr};
    real *energies {nullptr};
};

struct RVviewWithOldParticles : public RVview
{
    RVviewWithOldParticles(RodVector *rv, LocalRodVector *lrv);
    
    real4 *oldPositions {nullptr};
};

} // namespace mirheo
