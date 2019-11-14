#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/types/type_list.h>

namespace mirheo
{

namespace TypeAtomicAdd
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
__D__ inline void _addVect(T3 *addr, T3 s, real eps)
{
    _add(&addr->x, s.x, eps);
    _add(&addr->y, s.y, eps);
    _add(&addr->z, s.z, eps);
}



template <typename T>
__D__ inline void apply(__UNUSED T *addr, __UNUSED T s, __UNUSED const real eps = 0._r) {}

__D__ inline void apply(float  *addr, float  s, real eps = 0._r) {_add(addr, s, eps);}
__D__ inline void apply(double *addr, double s, real eps = 0._r) {_add(addr, s, eps);}
__D__ inline void apply(Force *addr, Force s,   real eps = 0._r) {_addVect(&addr->f, s.f, eps);}

__D__ inline void apply(Stress *addr, Stress s, real eps = 0._r)
{
    _add(&addr->xx, s.xx, eps);
    _add(&addr->xy, s.xy, eps);
    _add(&addr->xz, s.xz, eps);
    _add(&addr->yy, s.yy, eps);
    _add(&addr->yz, s.yz, eps);
    _add(&addr->zz, s.zz, eps);
}

__D__ inline void apply(RigidMotion *addr, RigidMotion s, real eps = 0._r)
{
    _addVect(&addr->force,  s.force,  eps);
    _addVect(&addr->torque, s.torque, eps);
}

} // namespace TypeAtomicAdd

} // namespace mirheo
