#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/type_map.h>

struct AtomicAdder
{
    AtomicAdder(float tolerance) :
        eps(tolerance)
    {}

    template <typename T>
    __D__ inline void operator()(T *addr, T var) const
    {
        add(addr, var);
    }

private:

    template <typename T>
    __D__ inline void add(T *addr, T s) const {}

    __D__ inline void add(float  *addr, float  s) const {_add(addr, s);}
    __D__ inline void add(double *addr, double s) const {_add(addr, s);}
    __D__ inline void add(Force *addr, Force s) const {_addVect(&addr->f, s.f);}

    __D__ inline void add(Stress *addr, Stress s) const
    {
        _add(&addr->xx, s.xx);
        _add(&addr->xy, s.xy);
        _add(&addr->xz, s.xz);
        _add(&addr->yy, s.yy);
        _add(&addr->yz, s.yz);
        _add(&addr->zz, s.zz);
    }

    __D__ inline void add(RigidMotion *addr, RigidMotion s) const
    {
        _addVect(&addr->force,  s.force);
        _addVect(&addr->torque, s.torque);
    }

    template <typename T>
    __D__ inline void _add(T *v, T s) const
    {
        if (fabs(s) >= eps) atomicAdd(v, s);
    }

    template <typename T3>
    __D__ inline void _addVect(T3 *addr, T3 s) const
    {
        _add(&addr->x, s.x);
        _add(&addr->y, s.y);
        _add(&addr->z, s.z);
    }
    
    const float eps;
};
