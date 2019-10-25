#pragma once

#include <core/datatypes.h>
#include <core/utils/vec_traits.h>

#ifdef MEMBRANE_FORCES_DOUBLE
using mReal  = double;
#else
using mReal  = float;
#endif // MEMBRANE_FORCES_DOUBLE

using mReal2 = VecTraits::Vec<mReal, 2>::Type;
using mReal3 = VecTraits::Vec<mReal, 3>::Type;


template<typename T3>
__D__ inline mReal3 make_mReal3(T3 v)
{
    return {v.x, v.y, v.z};
}

__D__ constexpr inline mReal3 make_mReal3(float a)
{
    return {static_cast<mReal>(a),
            static_cast<mReal>(a),
            static_cast<mReal>(a)};
}

__D__ constexpr inline mReal3 make_mReal3(double a)
{
    return {static_cast<mReal>(a),
            static_cast<mReal>(a),
            static_cast<mReal>(a)};
}

__D__ constexpr inline mReal operator "" _mr (const long double a)
{
    return static_cast<mReal>(a);
}

struct ParticleMReal
{
    mReal3 r, u;
};

template <typename View>
__D__ inline mReal3 fetchPosition(View view, int i)
{
    Real3_int ri(view.readPosition(i));
    return make_mReal3(ri.v);
}

template <typename View>
__D__ inline ParticleMReal fetchParticle(View view, int i)
{
    Particle p(view.readParticle(i));
    return {make_mReal3(p.r), make_mReal3(p.u)};
}
