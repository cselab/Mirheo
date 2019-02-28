#pragma once

#include <core/datatypes.h>

//#define RBC_FORCES_DOUBLE

#ifdef RBC_FORCES_DOUBLE
using real  = double;
using real3 = double3;
#else
using real  = float;
using real3 = float3;
#endif // RBC_FORCES_DOUBLE

template<typename T3>
__D__ inline real3 make_real3(T3 v)
{
    return {v.x, v.y, v.z};
}

__D__ constexpr inline real3 make_real3(float a)
{
    return {(real)a, (real)a, (real)a};
}

__D__ constexpr inline real3 make_real3(double a)
{
    return {(real)a, (real)a, (real)a};
}

__D__ constexpr inline real operator "" _r (const long double a)
{
    return (real) a;
}

struct ParticleReal
{
    real3 r, u;
};

template <typename View>
__D__ inline real3 fetchPosition(View view, int i)
{
    Particle p;
    p.readCoordinate(view.particles, i);
    return make_real3(p.r);
}

template <typename View>
__D__ inline ParticleReal fetchParticle(View view, int i)
{
    Particle p(view.particles, i);
    return {make_real3(p.r), make_real3(p.u)};
}
