#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

#ifndef MIRHEO_DOUBLE_PRECISION
static inline __HD__ RealRigidMotion toRealMotion(const DoubleRigidMotion& dm)
{
    RealRigidMotion sm;
    sm.r      = make_real3(dm.r);
    sm.q      = make_real4(dm.q);
    sm.vel    = make_real3(dm.vel);
    sm.omega  = make_real3(dm.omega);
    sm.force  = make_real3(dm.force);
    sm.torque = make_real3(dm.torque);
    return sm;
}
#endif

static inline __HD__ RealRigidMotion toRealMotion(const RealRigidMotion& sm)
{
    return sm;
}

template<class R3>
inline __HD__ RigidReal3 make_rigidReal3(R3 a)
{
    return {RigidReal(a.x), RigidReal(a.y), RigidReal(a.z)};
}

template<class R4>
inline __HD__ RigidReal4 make_rigidReal4(R4 a)
{
    return {RigidReal(a.x), RigidReal(a.y), RigidReal(a.z), RigidReal(a.w)};
}

} // namespace mirheo
