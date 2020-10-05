// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/vec_traits.h>

namespace mirheo
{

#ifdef MIRHEO_MEMBRANE_FORCES_DOUBLE
using mReal  = double; ///< precision switch for double precision membrane interactions
#else
using mReal  = float; ///< precision switch for single precision membrane interactions
#endif // MIRHEO_MEMBRANE_FORCES_DOUBLE

using mReal2 = vec_traits::Vec<mReal, 2>::Type; ///< real2 in mReal precision
using mReal3 = vec_traits::Vec<mReal, 3>::Type; ///< real3 in mReal precision

/// create a mReal3 from vector
template<typename T3>
__D__ inline mReal3 make_mReal3(T3 v)
{
    return {v.x, v.y, v.z};
}

/// create a mReal3 from scalar
__D__ constexpr inline mReal3 make_mReal3(float a)
{
    return {static_cast<mReal>(a),
            static_cast<mReal>(a),
            static_cast<mReal>(a)};
}

/// create a mReal3 from scalar
__D__ constexpr inline mReal3 make_mReal3(double a)
{
    return {static_cast<mReal>(a),
            static_cast<mReal>(a),
            static_cast<mReal>(a)};
}

inline namespace unit_literals {
__D__ constexpr inline mReal operator "" _mr (const long double a)
{
    return static_cast<mReal>(a);
}
} // namespace unit_literals

/// Simple particle in mReal precision
struct ParticleMReal
{
    mReal3 r; ///< position
    mReal3 u; ///< velocity
};

/// load position in \c mReal precision from a view
template <typename View>
__D__ inline mReal3 fetchPosition(View view, int i)
{
    Real3_int ri(view.readPosition(i));
    return make_mReal3(ri.v);
}

/// load ParticleMReal from a view
template <typename View>
__D__ inline ParticleMReal fetchParticle(View view, int i)
{
    Particle p(view.readParticle(i));
    return {make_mReal3(p.r), make_mReal3(p.u)};
}

} // namespace mirheo
