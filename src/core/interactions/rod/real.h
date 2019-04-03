#pragma once

#include <core/datatypes.h>

// #define ROD_FORCES_DOUBLE

#ifdef ROD_FORCES_DOUBLE
using real  = double;
using real3 = double3;
#else
using real  = float;
using real3 = float3;
#endif // ROD_FORCES_DOUBLE

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
    Float3_int ri(view.readPosition(i));
    return make_real3(ri.v);
}

template <typename View>
__D__ inline ParticleReal fetchParticle(View view, int i)
{
    Particle p(view.readParticle(i));
    return {make_real3(p.r), make_real3(p.u)};
}
