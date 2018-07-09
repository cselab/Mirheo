#pragma once

#if 1
using RigidReal  = double;
using RigidReal3 = double3;
using RigidReal4 = double4;

#else

using RigidReal  = float;
using RigidReal3 = float3;
using RigidReal4 = float4;
#endif

//=================================================================

template<class R3, class R4>
struct __align__(16) TemplRigidMotion
{
    R3 r;
    R4 q;
    R3 vel, omega;
    R3 force, torque;
};

using DoubleRigidMotion = TemplRigidMotion<double3, double4>;
using SingleRigidMotion = TemplRigidMotion<float3,  float4>;

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

using RigidMotion = TemplRigidMotion<RigidReal3, RigidReal4>;
