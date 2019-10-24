#pragma once

#include <core/datatypes.h>
#include <core/utils/vec_traits.h>

#ifdef ROD_FORCES_DOUBLE
using rReal  = double;
#else
using rReal  = float;
#endif // ROD_FORCES_DOUBLE

using rReal2 = VecTraits::Vec<rReal, 2>::Type;
using rReal3 = VecTraits::Vec<rReal, 3>::Type;
using rReal4 = VecTraits::Vec<rReal, 4>::Type;


template<typename T2>
__D__ inline rReal2 make_rReal2(T2 v)
{
    return {static_cast<rReal>(v.x),
            static_cast<rReal>(v.y)};
}

__D__ constexpr inline rReal2 make_rReal2(float a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

__D__ constexpr inline rReal2 make_rReal2(double a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

template<typename T3>
__D__ inline rReal3 make_rReal3(T3 v)
{
    return {static_cast<rReal>(v.x),
            static_cast<rReal>(v.y),
            static_cast<rReal>(v.z)};
}

__D__ constexpr inline rReal3 make_rReal3(float a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

__D__ constexpr inline rReal3 make_rReal3(double a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

template<typename T4>
__D__ inline rReal4 make_rReal4(T4 v)
{
    return {static_cast<rReal>(v.x),
            static_cast<rReal>(v.y),
            static_cast<rReal>(v.z),
            static_cast<rReal>(v.w)};
}

__D__ constexpr inline rReal4 make_rReal4(float a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

__D__ constexpr inline rReal4 make_rReal4(double a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

__D__ constexpr inline rReal operator "" _rr (const long double a)
{
    return static_cast<rReal>(a);
}

struct ParticleRReal
{
    rReal3 r, u;
};

template <typename View>
__D__ inline rReal3 fetchPosition(View view, int i)
{
    Float3_int ri(view.readPosition(i));
    return make_rReal3(ri.v);
}

template <typename View>
__D__ inline ParticleRReal fetchParticle(View view, int i)
{
    Particle p(view.readParticle(i));
    return {make_rReal3(p.r), make_rReal3(p.u)};
}
