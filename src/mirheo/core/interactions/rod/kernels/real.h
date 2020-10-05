// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/vec_traits.h>

namespace mirheo
{

#ifdef MIRHEO_ROD_FORCES_DOUBLE
using rReal  = double; ///< double precision switch
#else
using rReal  = float; ///< single precision switch
#endif // MIRHEO_ROD_FORCES_DOUBLE

using rReal2 = vec_traits::Vec<rReal, 2>::Type; ///< real2
using rReal3 = vec_traits::Vec<rReal, 3>::Type; ///< real3
using rReal4 = vec_traits::Vec<rReal, 4>::Type; ///< real4

/// create real2 from vector
template<typename T2>
__D__ inline rReal2 make_rReal2(T2 v)
{
    return {static_cast<rReal>(v.x),
            static_cast<rReal>(v.y)};
}

/// create real2 from scalar
__D__ constexpr inline rReal2 make_rReal2(float a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

/// create real2 from scalar
__D__ constexpr inline rReal2 make_rReal2(double a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

/// create real3 from vector
template<typename T3>
__D__ inline rReal3 make_rReal3(T3 v)
{
    return {static_cast<rReal>(v.x),
            static_cast<rReal>(v.y),
            static_cast<rReal>(v.z)};
}

/// create real3 from scalar
__D__ constexpr inline rReal3 make_rReal3(float a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

/// create real3 from scalar
__D__ constexpr inline rReal3 make_rReal3(double a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

/// create real4 from vector
template<typename T4>
__D__ inline rReal4 make_rReal4(T4 v)
{
    return {static_cast<rReal>(v.x),
            static_cast<rReal>(v.y),
            static_cast<rReal>(v.z),
            static_cast<rReal>(v.w)};
}

/// create real4 from scalar
__D__ constexpr inline rReal4 make_rReal4(float a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

/// create real4 from scalar
__D__ constexpr inline rReal4 make_rReal4(double a)
{
    return {static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a),
            static_cast<rReal>(a)};
}

inline namespace unit_literals {
    __D__ constexpr inline rReal operator "" _rr (const long double a)
    {
        return static_cast<rReal>(a);
    }
} // namespace unit_literals

/// particle structure in required precision
struct ParticleRReal
{
    rReal3 r; ///< position
    rReal3 u; ///< velocity
};

/// read position in required precision from a view
template <typename View>
__D__ inline rReal3 fetchPosition(View view, int i)
{
    Real3_int ri(view.readPosition(i));
    return make_rReal3(ri.v);
}

/// read position and velocity in required precision from a view
template <typename View>
__D__ inline ParticleRReal fetchParticle(View view, int i)
{
    Particle p(view.readParticle(i));
    return {make_rReal3(p.r), make_rReal3(p.u)};
}

} // namespace mirheo
