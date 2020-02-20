#pragma once

#include "ov.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/rigid/rigid_motion.h>

namespace mirheo
{

class RigidObjectVector;
class LocalRigidObjectVector;

/// A \c OVview with additional rigid object infos
struct ROVview : public OVview
{
    /** \brief Construct a \c ROVview 
        \param [in] rov The RigidObjectVector that the view represents
        \param [in] lrov The LocalRigidObjectVector that the view represents
    */
    ROVview(RigidObjectVector *rov, LocalRigidObjectVector *lrov);
    
    RigidMotion *motions {nullptr}; ///< rigid object states

    real3 J   {0._r, 0._r, 0._r}; ///< diagonal entries of inertia tensor
    real3 J_1 {0._r ,0._r, 0._r}; ///< diagonal entries of the inverse inertia tensor
};

/// A \c OVview with additional rigid object info from previous time step
struct ROVviewWithOldMotion : public ROVview
{
    /** \brief Construct a \c ROVview 
        \param [in] rov The RigidObjectVector that the view represents
        \param [in] lrov The LocalRigidObjectVector that the view represents

        \rst
        .. warning::
            The rov must hold old motions channel.
        \endrst
    */
    ROVviewWithOldMotion(RigidObjectVector* rov, LocalRigidObjectVector* lrov);
    
    RigidMotion *old_motions {nullptr}; ///< rigid object states at previous time step
};

} // namespace mirheo
