// Copyright 2020 ETH Zurich. All Rights Reserved.
/*
 *  dpd-rng.h
 *  Part of Mirheo/cuda-dpd-sem/
 *
 *  Created and authored by Diego Rossinelli on 2015-02-12.
 *  Major editing (+Logistic RNG) from Yu-Hang Tang on 2015-03-19.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */
#pragma once

#include <cstdlib>
#include <limits>
#include <stdint.h>
#include <cmath>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

#ifndef __NVCC__
/// fused multiply - add, single precision
float __fmaf_rz(float x, float y, float z)
{
    return x*y + z;
}
/// fused multiply - add, double precision
double __fma_rz(double x, double y, double z)
{
    return x*y + z;
}
#endif


namespace Logistic
{
__D__ real mean0var1( real seed, uint i, uint j );
__D__ real mean0var1( real seed, int i, int j );
__D__ real mean0var1( real seed, real i, real j );
}

namespace Logistic
{
/************************* Branch generator **********************
 * Make one random number per pair of particles per timestep
 * Based on the Logistic map on interval [-1,1]
 * each branch no weaker than trunk
 * zero dependency between branches
 *****************************************************************/

// floating point version of LCG
inline __D__ real rem( real r ) {
    return r - math::floor( r );
}

// FMA wrapper for the convenience of switching rouding modes
inline __D__ float FMA( float x, float y, float z ) {
    return __fmaf_rz( x, y, z );
}
inline __D__ double FMA( double x, double y, double z ) {
    return __fma_rz( x, y, z );
}

// logistic rounds
// <3> : 4 FMA + 1 MUL
// <2> : 2 FMA + 1 MUL
// <1> : 1 FMA + 1 MUL
template<int N> inline __D__ real __logistic_core( real x )
{
    real x2 = x * x;
    real r = FMA( FMA( 8.0_r, x2, -8.0_r ), x2, 1.0_r );
    return __logistic_core < N - 2 > ( r );
}

template<> inline __D__ real __logistic_core<1>( real x ) {
    return FMA( 2.0_r * x, x, -1.0_r );
}

template<> inline __D__ real __logistic_core<0>( real x ) {
    return x;
}

// random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
// mean = 0
// variance = 1
// can be used directly for DPD

// passes of logistic map
constexpr int N = 18;
// spacing coefficients for low discrepancy numbers
constexpr real gold       = 0.6180339887498948482_r;
constexpr real hugegold   = 0.6180339887498948482E39_r;
constexpr real silver     = 0.4142135623730950488_r;
constexpr real hugesilver = 0.4142135623730950488E39_r;
constexpr real bronze     = 0.00008877875787352212838853023_r;
constexpr real tin        = 0.00004602357186447026756768986_r;
// square root of 2
constexpr real sqrt2 = 1.41421356237309514547_r;

inline __D__ real uniform01( real seed, int i, int j )
{
    const real val = mean0var1(seed, i, j) * (0.5_r/sqrt2) + 0.5_r;
    return math::max(0.0_r, math::min(1.0_r, val));
}

inline __D__ real mean0var1( real seed, int u, int v )
{
    const real p = rem( ( ( u & 0x3FF ) * gold ) + u * bronze + ( ( v & 0x3FF ) * silver ) + v * tin ); // safe for large u or v
    const real q = rem( seed );
    const real l = __logistic_core<N>( q - p );
    return l * sqrt2;
}

inline __D__ real mean0var1( real seed, uint u, uint v )
{
    // 7 FLOPS
    const real p = rem( ( ( u & 0x3FFU ) * gold ) + u * bronze + ( ( v & 0x3FFU ) * silver ) + v * tin ); // safe for large u or v
    const real q = rem( seed );

    // 45+1 FLOPS
    const real l = __logistic_core<N>( q - p );
    // 1 FLOP
    return l * sqrt2;
}

inline __D__ real mean0var1( real seed, real u, real v )
{
    const real p = rem( math::sqrt(u) * gold + sqrtf(v) * silver );
    const real q = rem( seed );

    const real l = __logistic_core<N>( q - p );
    return l * sqrt2;
}

inline __D__ real mean0var1_dual( real seed, real u, real v )
{
    const real p = rem( math::sqrt(u) * gold + sqrtf(v) * silver );
    const real q = rem( seed );

    const real l = __logistic_core<N>( q - p );
    const real z = __logistic_core<N>( q + p - 1._r );
    return l + z;
}
} // namespace Logistic

namespace Saru
{
__D__ real mean0var1( real seed, uint i, uint j );
__D__ real mean0var1( real seed, int i, int j );
__D__ real mean0var1( real seed, real i, real j );
__D__ real uniform01( real seed, uint i, uint j );
__D__ real2 normal2( real seed, uint i, uint j );
}

namespace Saru
{
__D__ inline real saru( unsigned int seed1, unsigned int seed2, unsigned int seed3 )
{
    seed3 ^= ( seed1 << 7 ) ^ ( seed2 >> 6 );
    seed2 += ( seed1 >> 4 ) ^ ( seed3 >> 15 );
    seed1 ^= ( seed2 << 9 ) + ( seed3 << 8 );
    seed3 ^= 0xA5366B4D * ( ( seed2 >> 11 ) ^ ( seed1 << 1 ) );
    seed2 += 0x72BE1579 * ( ( seed1 << 4 )  ^ ( seed3 >> 16 ) );
    seed1 ^= 0X3F38A6ED * ( ( seed3 >> 5 )  ^ ( ( ( signed int )seed2 ) >> 22 ) );
    seed2 += seed1 * seed3;
    seed1 += seed3 ^ ( seed2 >> 2 );
    seed2 ^= ( ( signed int )seed2 ) >> 17;

    int state  = 0x79dedea3 * ( seed1 ^ ( ( ( signed int )seed1 ) >> 14 ) );
    int wstate = ( state + seed2 ) ^ ( ( ( signed int )state ) >> 8 );
    state  = state + ( wstate * ( wstate ^ 0xdddf97f5 ) );
    wstate = 0xABCB96F7 + ( wstate >> 1 );

    state  = 0x4beb5d59 * state + 0x2600e1f7; // LCG
    wstate = wstate + 0x8009d14b + ( ( ( ( signed int )wstate ) >> 31 ) & 0xda879add ); // OWS

    const unsigned int v = ( state ^ ( state >> 26 ) ) + wstate;
    const unsigned int r = ( v ^ ( v >> 20 ) ) * 0x6957f5a7;

    const real res = r / ( 4294967295.0_r );
    return res;
}

inline __D__ real2 normal2( real seed, uint i, uint j )
{
    const real u1 = uniform01( seed, math::min(i, j),   math::max(i, j) );
    const real u2 = uniform01( u1,   math::max(i, j)+1, math::min(i, j) );

    const real r = math::sqrt(-2.0_r * logf(u1));
    const real theta = 2.0_r * static_cast<real>(M_PI) * u2;

    auto res = math::sincos(theta);
    res *= r;

    return res;
}

inline __D__ real uniform01( real seed, uint i, uint j )
{
    auto t = reinterpret_cast<unsigned int*>(&seed);
    unsigned int tag = *t;

    return saru( tag, i, j );
}

inline __D__ real mean0var1( real seed, uint i, uint j )
{
    return uniform01(seed, i, j) * 3.464101615_r - 1.732050807_r;
}

inline __D__ real mean0var1( real seed, int i, int j )
{
    return mean0var1( seed, (uint) i, (uint) j );
}

inline __D__ real mean0var1( real seed, real i, real j )
{
    return mean0var1( seed, (uint) i, (uint) j );
}

} // namespace Saru

} // namespace mirheo
