// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "object_belonging.h"

namespace mirheo
{
/** \brief Check in/out status of particles against a RigidShapedObjectVector.
    \tparam Shape The AnalyticShape that represent the shape of the objects.
 */
template <class Shape>
class ShapeBelongingChecker : public ObjectVectorBelongingChecker
{
public:
    using ObjectVectorBelongingChecker::ObjectVectorBelongingChecker;

protected:
    void _tagInner(ParticleVector *pv, CellList *cl, cudaStream_t stream) override;
};

} // namespace mirheo
