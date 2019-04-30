#pragma once

#include "../rigid_ellipsoid_object_vector.h"
#include "rov.h"

/**
 * GPU-compatible struct of all the relevant data
 */
struct REOVview : public ROVview
{
    float3 axes    = {0,0,0};
    float3 invAxes = {0,0,0};

    REOVview(RigidEllipsoidObjectVector* reov = nullptr, LocalRigidObjectVector* lrov = nullptr) :
        ROVview(reov, lrov)
    {
        if (reov == nullptr || lrov == nullptr) return;

        // More fields
        axes = reov->axes;
        invAxes = 1.0 / axes;
    }
};

struct REOVviewWithOldMotion : public REOVview
{
    RigidMotion *old_motions = nullptr;

    REOVviewWithOldMotion(RigidEllipsoidObjectVector* reov = nullptr, LocalRigidObjectVector* lrov = nullptr) :
        REOVview(reov, lrov)
    {
        if (reov == nullptr || lrov == nullptr) return;

        old_motions = lrov->dataPerObject.getData<RigidMotion>(ChannelNames::oldMotions)->devPtr();
    }
};
