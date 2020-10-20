// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/types/type_list.h>

namespace mirheo
{

namespace type_atomic_add
{

namespace details
{

template <typename T>
__D__ inline void _add(T *v, T s, real eps)
{
    if (math::abs(s) >= eps)
    {
#ifdef __CUDACC__
        atomicAdd(v, s);
#else
        *v += s;
#endif // __CUDACC__
    }
}

template <typename T3>
__D__ inline void _addVect3(T3 *addr, T3 s, real eps)
{
    _add(&addr->x, s.x, eps);
    _add(&addr->y, s.y, eps);
    _add(&addr->z, s.z, eps);
}

} // namespace details


template <typename T>
__D__ inline void apply(__UNUSED T *addr, __UNUSED T s, __UNUSED const real eps = 0._r) {}

__D__ inline void apply(float  *addr, float  s, real eps = 0._r) {details::_add(addr, s, eps);}
__D__ inline void apply(double *addr, double s, real eps = 0._r) {details::_add(addr, s, eps);}
__D__ inline void apply(Force  *addr, Force  s, real eps = 0._r) {details::_addVect3(&addr->f, s.f, eps);}

__D__ inline void apply(Stress *addr, Stress s, real eps = 0._r)
{
    details::_add(&addr->xx, s.xx, eps);
    details::_add(&addr->xy, s.xy, eps);
    details::_add(&addr->xz, s.xz, eps);
    details::_add(&addr->yy, s.yy, eps);
    details::_add(&addr->yz, s.yz, eps);
    details::_add(&addr->zz, s.zz, eps);
}

__D__ inline void apply(RigidMotion *addr, RigidMotion s, real eps = 0._r)
{
    details::_addVect3(&addr->force,  s.force,  eps);
    details::_addVect3(&addr->torque, s.torque, eps);
}

} // namespace type_atomic_add





namespace type_add
{

namespace details
{

template <typename T>
__D__ inline void _add(T *v, T s)
{
    *v += s;
}

template <typename T3>
__D__ inline void _addVect3(T3 *addr, T3 s)
{
    _add(&addr->x, s.x);
    _add(&addr->y, s.y);
    _add(&addr->z, s.z);
}

} // namespace details


template <typename T>
__D__ inline void apply(__UNUSED T *addr, __UNUSED T s) {}

__D__ inline void apply(float  *addr, float  s) {details::_add(addr, s);}
__D__ inline void apply(double *addr, double s) {details::_add(addr, s);}
__D__ inline void apply(Force  *addr, Force  s) {details::_addVect3(&addr->f, s.f);}

__D__ inline void apply(Stress *addr, Stress s)
{
    details::_add(&addr->xx, s.xx);
    details::_add(&addr->xy, s.xy);
    details::_add(&addr->xz, s.xz);
    details::_add(&addr->yy, s.yy);
    details::_add(&addr->yz, s.yz);
    details::_add(&addr->zz, s.zz);
}

__D__ inline void apply(RigidMotion *addr, RigidMotion s)
{
    details::_addVect3(&addr->force,  s.force);
    details::_addVect3(&addr->torque, s.torque);
}

} // namespace type_add



namespace type_scale
{

namespace details
{

template <typename T>
__D__ inline void _scale(T *v, real s)
{
    *v *= s;
}

template <typename T3>
__D__ inline void _scaleVect3(T3 *addr, real s)
{
    _scale(&addr->x, s);
    _scale(&addr->y, s);
    _scale(&addr->z, s);
}

} // namespace details


template <typename T>
__D__ inline void apply(__UNUSED T *addr, __UNUSED real s) {}

__D__ inline void apply(float  *addr, real s) {details::_scale(addr, s);}
__D__ inline void apply(double *addr, real s) {details::_scale(addr, s);}
__D__ inline void apply(Force  *addr, real s) {details::_scaleVect3(&addr->f, s);}

__D__ inline void apply(Stress *addr, real s)
{
    details::_scale(&addr->xx, s);
    details::_scale(&addr->xy, s);
    details::_scale(&addr->xz, s);
    details::_scale(&addr->yy, s);
    details::_scale(&addr->yz, s);
    details::_scale(&addr->zz, s);
}

__D__ inline void apply(RigidMotion *addr, real s)
{
    details::_scaleVect3(&addr->force,  s);
    details::_scaleVect3(&addr->torque, s);
}

} // namespace type_scale

} // namespace mirheo
