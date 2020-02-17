#pragma once

#include "object_belonging.h"

namespace mirheo
{

template <class Shape>
class ShapeBelongingChecker : public ObjectVectorBelongingChecker
{
public:
    using ObjectVectorBelongingChecker::ObjectVectorBelongingChecker;

protected:
    void tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;
};

} // namespace mirheo
