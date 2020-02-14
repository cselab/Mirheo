#pragma once

#include "rov.h"
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>

namespace mirheo
{

template <class Shape>
struct RSOVview : public ROVview
{
    RSOVview(RigidShapedObjectVector<Shape> *rsov, LocalRigidObjectVector *lrov) :
        ROVview(rsov, lrov),
        shape(rsov->getShape())
    {}

    Shape shape;
};

template <class Shape>
struct RSOVviewWithOldMotion : public RSOVview<Shape>
{
    RSOVviewWithOldMotion(RigidShapedObjectVector<Shape> *rsov, LocalRigidObjectVector *lrov) :
        RSOVview<Shape>(rsov, lrov)
    {
        old_motions = lrov->dataPerObject.getData<RigidMotion>(ChannelNames::oldMotions)->devPtr();
    }

    RigidMotion *old_motions {nullptr};
};

} // namespace mirheo
