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
__D__ inline void _add(T *v, T s)
{
    if (math::abs(s) > 0)
    {
#ifdef __CUDACC__
        atomicAdd(v, s);
#else
        *v += s;
#endif // __CUDACC__
    }
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

} // namespace type_atomic_add


} // namespace mirheo
