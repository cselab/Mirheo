#pragma once

#include <core/datatypes.h>

#define ROD_FORCES_DOUBLE

#ifdef ROD_FORCES_DOUBLE
using real  = double;
using real2 = double2;
using real3 = double3;
#else
using real  = float;
using real2 = float2;
using real3 = float3;
#endif // ROD_FORCES_DOUBLE

template<typename T2>
__D__ inline real2 make_real2(T2 v)
{
    return {(real) v.x, (real) v.y};
}

template<typename T3>
__D__ inline real3 make_real3(T3 v)
{
    return {(real) v.x, (real) v.y, (real) v.z};
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
