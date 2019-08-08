#pragma once

#include "../rigid_object_vector.h"
#include "ov.h"

/**
 * GPU-compatible struct of all the relevant data
 */
struct ROVview : public OVview
{
    RigidMotion *motions = nullptr;

    float3 J   {0,0,0};
    float3 J_1 {0,0,0};

    ROVview(RigidObjectVector *rov = nullptr, LocalRigidObjectVector *lrov = nullptr) :
        OVview(rov, lrov)
    {
        if (rov == nullptr || lrov == nullptr) return;

        motions = lrov->dataPerObject.getData<RigidMotion>(ChannelNames::motions)->devPtr();

        J   = rov->J;
        J_1 = 1.0 / J;
    }
};

struct ROVviewWithOldMotion : public ROVview
{
    RigidMotion *old_motions = nullptr;

    ROVviewWithOldMotion(RigidObjectVector* rov = nullptr, LocalRigidObjectVector* lrov = nullptr) :
        ROVview(rov, lrov)
    {
        if (rov == nullptr || lrov == nullptr) return;

        old_motions = lrov->dataPerObject.getData<RigidMotion>(ChannelNames::oldMotions)->devPtr();
    }
};

