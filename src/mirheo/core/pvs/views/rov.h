#pragma once

#include "../rigid_object_vector.h"
#include "ov.h"

namespace mirheo
{

/**
 * GPU-compatible struct of all the relevant data
 */
struct ROVview : public OVview
{
    RigidMotion *motions {nullptr};

    real3 J   {0._r, 0._r, 0._r};
    real3 J_1 {0._r ,0._r, 0._r};

    ROVview(RigidObjectVector *rov, LocalRigidObjectVector *lrov) :
        OVview(rov, lrov)
    {
        motions = lrov->dataPerObject.getData<RigidMotion>(ChannelNames::motions)->devPtr();

        J   = rov->J;
        J_1 = 1.0 / J;
    }
};

struct ROVviewWithOldMotion : public ROVview
{
    RigidMotion *old_motions {nullptr};

    ROVviewWithOldMotion(RigidObjectVector* rov, LocalRigidObjectVector* lrov) :
        ROVview(rov, lrov)
    {
        old_motions = lrov->dataPerObject.getData<RigidMotion>(ChannelNames::oldMotions)->devPtr();
    }
};

} // namespace mirheo
