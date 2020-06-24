// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "object_belonging.h"

namespace mirheo
{
/// \brief Check in/out status of particles against an ObjectVector with a triangle mesh.
class MeshBelongingChecker : public ObjectVectorBelongingChecker
{
public:
    using ObjectVectorBelongingChecker::ObjectVectorBelongingChecker;

protected:
    void _tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;
};

} // namespace mirheo
