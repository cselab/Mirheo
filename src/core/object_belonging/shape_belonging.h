#pragma once

#include "object_belonging.h"

template <class Shape>
class ShapeBelongingChecker : public ObjectBelongingChecker_Common
{
public:
    using ObjectBelongingChecker_Common::ObjectBelongingChecker_Common;

    void tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;
};
