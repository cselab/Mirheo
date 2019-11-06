#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/quaternion.h>

#define RIGID_MOTIONS_DOUBLE

namespace mirheo
{

#ifdef RIGID_MOTIONS_DOUBLE
using RigidReal  = double;
#else
using RigidReal  = float;
#endif

using RigidReal3 = VecTraits::Vec<RigidReal, 3>::Type;
using RigidReal4 = VecTraits::Vec<RigidReal, 4>::Type;
using RigiQuaternion = Quaternion<RigidReal>;

template <class RealType>
struct __align__(16) TemplRigidMotion
{
    using R3 = typename VecTraits::Vec<RealType, 3>::Type;
    
    R3 r;
    Quaternion<RealType> q;
    R3 vel, omega;
    R3 force, torque;
};

using DoubleRigidMotion = TemplRigidMotion<double>;
using RealRigidMotion   = TemplRigidMotion<real>;
using RigidMotion       = TemplRigidMotion<RigidReal>;

} // namespace mirheo
