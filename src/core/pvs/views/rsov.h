#pragma once

#include "rov.h"
#include <core/pvs/rigid_ashape_object_vector.h>

template <class Shape>
struct RSOVview : public ROVview
{
    Shape shape;
    
    RSOVview(RigidShapedObjectVector<Shape>* rsov = nullptr, LocalRigidObjectVector* lrov = nullptr) :
        ROVview(rsov, lrov),
        shape(rsov->shape)
    {}
};

template <class Shape>
struct RSOVviewWithOldMotion : public RSOVview<Shape>
{
    RigidMotion *old_motions = nullptr;

    RSOVviewWithOldMotion(RigidShapedObjectVector<Shape>* rsov = nullptr, LocalRigidObjectVector* lrov = nullptr) :
        RSOVview<Shape>(rsov, lrov)
    {
        if (rsov == nullptr || lrov == nullptr) return;

        old_motions = lrov->dataPerObject.getData<RigidMotion>(ChannelNames::oldMotions)->devPtr();
    }
};
