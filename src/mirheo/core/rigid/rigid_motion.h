#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/quaternion.h>

#define RIGID_MOTIONS_DOUBLE

namespace mirheo
{

#ifdef RIGID_MOTIONS_DOUBLE
using RigidReal = double; ///< precision used for rigid states
#else
using RigidReal = float;  ///< precision used for rigid states
#endif

using RigidReal3 = VecTraits::Vec<RigidReal, 3>::Type; ///< real3
using RigidReal4 = VecTraits::Vec<RigidReal, 4>::Type; ///< real4
using RigiQuaternion = Quaternion<RigidReal>; ///< quaternion

/** \brief Holds the state of a rigid object

    A rigid object state is defined by its position and orientation as 
    well as linear and angular velocities.
    For convenience, the force and torque are also stored in this structure.
 */
template <class RealType>
struct __align__(16) TemplRigidMotion
{
    /// real3
    using R3 = typename VecTraits::Vec<RealType, 3>::Type;
    
    R3 r;                   ///< position of the center of mass
    Quaternion<RealType> q; ///< orientation
    R3 vel;                 ///< linear velocity
    R3 omega;               ///< angular velocity
    R3 force;               ///< force
    R3 torque;              ///< torque
};

using DoubleRigidMotion = TemplRigidMotion<double>;    ///< Rigid state in double precision
using RealRigidMotion   = TemplRigidMotion<real>;      ///< Rigid state in real precision
using RigidMotion       = TemplRigidMotion<RigidReal>; ///< Rigid state in RigidReal precision

} // namespace mirheo
