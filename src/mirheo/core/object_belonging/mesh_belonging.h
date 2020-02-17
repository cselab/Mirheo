#pragma once

#include "object_belonging.h"

namespace mirheo
{

class MeshBelongingChecker : public ObjectVectorBelongingChecker
{
public:
    using ObjectVectorBelongingChecker::ObjectVectorBelongingChecker;

    void tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;
};

} // namespace mirheo
