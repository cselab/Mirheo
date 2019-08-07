#pragma once

#include <core/datatypes.h>

inline __host__ __device__ SingleRigidMotion toSingleMotion(DoubleRigidMotion dm)
{
    SingleRigidMotion sm;
    sm.r      = make_float3(dm.r);
    sm.q      = make_float4(dm.q);
    sm.vel    = make_float3(dm.vel);
    sm.omega  = make_float3(dm.omega);
    sm.force  = make_float3(dm.force);
    sm.torque = make_float3(dm.torque);

    return sm;
}

inline __host__ __device__ SingleRigidMotion toSingleMotion(SingleRigidMotion sm)
{
    return sm;
}

template<class R3>
inline __host__ __device__ RigidReal3 make_rigidReal3(R3 a)
{
    return {RigidReal(a.x), RigidReal(a.y), RigidReal(a.z)};
}

template<class R4>
inline __host__ __device__ RigidReal4 make_rigidReal4(R4 a)
{
    return {RigidReal(a.x), RigidReal(a.y), RigidReal(a.z), RigidReal(a.w)};
}
