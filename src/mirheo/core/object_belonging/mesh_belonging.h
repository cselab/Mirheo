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

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    void _tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;
};

} // namespace mirheo
