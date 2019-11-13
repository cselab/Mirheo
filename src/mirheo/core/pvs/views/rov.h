#pragma once

#include "ov.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/rigid/rigid_motion.h>

namespace mirheo
{

class RigidObjectVector;
class LocalRigidObjectVector;

struct ROVview : public OVview
{
    ROVview(RigidObjectVector *rov, LocalRigidObjectVector *lrov);
    
    RigidMotion *motions {nullptr};

    real3 J   {0._r, 0._r, 0._r};
    real3 J_1 {0._r ,0._r, 0._r};
};

struct ROVviewWithOldMotion : public ROVview
{
    ROVviewWithOldMotion(RigidObjectVector* rov, LocalRigidObjectVector* lrov);
    
    RigidMotion *old_motions {nullptr};
};

} // namespace mirheo
