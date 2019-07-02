#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/type_map.h>

namespace TypeAtomicAdd
{

template <typename T>
__D__ inline void _add(T *v, T s, float eps)
{
    if (fabs(s) >= eps)
    {
#ifdef __CUDACC__
        atomicAdd(v, s);
#else
        *v += s;
#endif // __CUDACC__
    }
}

template <typename T3>
__D__ inline void _addVect(T3 *addr, T3 s, float eps)
{
    _add(&addr->x, s.x, eps);
    _add(&addr->y, s.y, eps);
    _add(&addr->z, s.z, eps);
}



template <typename T>
__D__ inline void apply(T *addr, T s, const float eps = 0.f) {}

__D__ inline void apply(float  *addr, float  s, float eps = 0.f) {_add(addr, s, eps);}
__D__ inline void apply(double *addr, double s, float eps = 0.f) {_add(addr, s, eps);}
__D__ inline void apply(Force *addr, Force s,   float eps = 0.f) {_addVect(&addr->f, s.f, eps);}

__D__ inline void apply(Stress *addr, Stress s, float eps = 0.f)
{
    _add(&addr->xx, s.xx, eps);
    _add(&addr->xy, s.xy, eps);
    _add(&addr->xz, s.xz, eps);
    _add(&addr->yy, s.yy, eps);
    _add(&addr->yz, s.yz, eps);
    _add(&addr->zz, s.zz, eps);
}

__D__ inline void apply(RigidMotion *addr, RigidMotion s, float eps = 0.f)
{
    _addVect(&addr->force,  s.force,  eps);
    _addVect(&addr->torque, s.torque, eps);
}

} // namespace TypeAtomicAdd
