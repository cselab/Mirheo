/*
 *  dpd-rng.h
 *  Part of uDeviceX/cuda-dpd-sem/
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

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#ifndef __NVCC__
float __fmaf_rz(float x, float y, float z)
{
    return x*y + z;
}
#endif


namespace Logistic
{
    __D__ float mean0var1( float seed, uint i, uint j );
    __D__ float mean0var1( float seed, int i, int j );
    __D__ float mean0var1( float seed, float i, float j );
}

namespace Logistic
{
    /************************* Trunk generator ***********************
     * Make one global random number per each timestep
     * cite G. Marsaglia
     * passes BigCrush
     *****************************************************************/
    struct KISS {
        typedef uint32_t integer;
        integer x, y, z, c;

        KISS() : x( 0 ), y( 0 ), z( 0 ), c( 0 ) {}

        KISS( integer x_, integer y_, integer z_, integer c_ ) :
            x( x_ ), y( y_ ), z( z_ ), c( c_ ) {}

        float get_float()
        {
            return get_int() / float( std::numeric_limits<integer>::max() );
        }

        integer get_int()
        {
            uint64_t t, a = 698769069ULL;
            x = 69069 * x + 12345;
            y ^= ( y << 13 );
            y ^= ( y >> 17 );
            y ^= ( y << 5 ); /* y must never be set to zero! */
            t = a * z + c;
            c = ( t >> 32 ); /* Also avoid setting z=c=0! */
            return x + y + ( z = t );
        }
    };

#ifdef __CUDACC__

    /************************* Branch generator **********************
     * Make one random number per pair of particles per timestep
     * Based on the Logistic map on interval [-1,1]
     * each branch no weaker than trunk
     * zero dependency between branches
     *****************************************************************/

    // floating point version of LCG
    inline __D__ float rem( float r ) {
        return r - floorf( r );
    }

    // FMA wrapper for the convenience of switching rouding modes
    inline __D__ float FMA( float x, float y, float z ) {
        return __fmaf_rz( x, y, z );
    }

    // logistic rounds
    // <3> : 4 FMA + 1 MUL
    // <2> : 2 FMA + 1 MUL
    // <1> : 1 FMA + 1 MUL
#if 0
    template<int N> inline __D__ float __logistic_core( float x ) {
        float x2;
        // saturated square
        // clamp result to [0,1]
        // to cancel error introduced by logistic<3> \in [-1.000001, 1.000003]
        asm("mul.f32.sat %0, %1, %1;" : "=f"(x2) : "f"(x), "f"(x) );
        float r = FMA( FMA( FMA( FMA( 128.0, x2, -256.0 ), x2, 160.0 ), x2, -32.0 ), x2, 1.0 );
        return __logistic_core < N - 3 > ( r );
    }
    template<int N> struct __logistic_core_flops_counter {
        const static unsigned long long FLOPS = 9 + __logistic_core_flops_counter<N-3>::FLOPS;
    };

    template<> inline __D__ float __logistic_core<2>( float x ) {
        float x2 = x * x;
        return FMA( FMA( 8.0, x2, -8.0 ), x2, 1.0 );
    }
    template<> struct __logistic_core_flops_counter<2> {
        const static unsigned long long FLOPS = 5;
    };

#else
    template<int N> inline __D__ float __logistic_core( float x )
    {
        float x2 = x * x;
        float r = FMA( FMA( 8.0, x2, -8.0 ), x2, 1.0 );
        return __logistic_core < N - 2 > ( r );
    }
    template<int N> struct __logistic_core_flops_counter {
        const static unsigned long long FLOPS = 5 + __logistic_core_flops_counter<N-2>::FLOPS;
    };
#endif

    template<> inline __D__ float __logistic_core<1>( float x ) {
        return FMA( 2.0 * x, x, -1.0 );
    }
    template<> struct __logistic_core_flops_counter<1> {
        const static unsigned long long FLOPS = 3;
    };

    template<> inline __D__ float __logistic_core<0>( float x ) {
        return x;
    }
    template<> struct __logistic_core_flops_counter<0> {
        const static unsigned long long FLOPS = 0;
    };

    // random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
    // mean = 0
    // variance = 1
    // can be used directly for DPD

    // passes of logistic map
    const static int N = 18;
    // spacing coefficints for low discrepancy numbers
    const static float gold   = 0.6180339887498948482;
    const static float hugegold   = 0.6180339887498948482E39;
    const static float silver = 0.4142135623730950488;
    const static float hugesilver = 0.4142135623730950488E39;
    const static float bronze = 0.00008877875787352212838853023;
    const static float tin    = 0.00004602357186447026756768986;
    // square root of 2
    const static float sqrt2 = 1.41421356237309514547;

    inline __D__ float uniform01( float seed, int i, int j )
    {
        float val = mean0var1(seed, i, j) * (0.5f/sqrt2) + 0.5f;
        return max(0.0f, min(1.0f, val));
    }

    inline __D__ float mean0var1( float seed, int u, int v )
    {
        float p = rem( ( ( u & 0x3FF ) * gold ) + u * bronze + ( ( v & 0x3FF ) * silver ) + v * tin ); // safe for large u or v
        float q = rem( seed );
        float l = __logistic_core<N>( q - p );
        return l * sqrt2;
    }

    inline __D__ float mean0var1( float seed, uint u, uint v )
    {
        // 7 FLOPS
        float p = rem( ( ( u & 0x3FFU ) * gold ) + u * bronze + ( ( v & 0x3FFU ) * silver ) + v * tin ); // safe for large u or v
        float q = rem( seed );

        // 45+1 FLOPS
        float l = __logistic_core<N>( q - p );
        // 1 FLOP
        return l * sqrt2;
    }
    struct mean0var1_flops_counter {
        const static unsigned long long FLOPS = 9ULL + __logistic_core_flops_counter<N>::FLOPS;
    };

    inline __D__ float mean0var1( float seed, float u, float v )
    {
        float p = rem( sqrtf(u) * gold + sqrtf(v) * silver ); // Acknowledging Dmitry for the use of sqrtf
        float q = rem( seed );

        float l = __logistic_core<N>( q - p );
        return l * sqrt2;
    }

    inline __D__ float mean0var1_dual( float seed, float u, float v )
    {
        float p = rem( sqrtf(u) * gold + sqrtf(v) * silver ); // Acknowledging Dmitry for the use of sqrtf
        float q = rem( seed );

        float l = __logistic_core<N>( q - p );
        float z = __logistic_core<N>( q + p - 1.f );
        return l + z;
    }
}

namespace Saru
{
    __D__ float mean0var1( float seed, uint i, uint j );
    __D__ float mean0var1( float seed, int i, int j );
    __D__ float mean0var1( float seed, float i, float j );
    __D__ float uniform01( float seed, uint i, uint j );
    __D__ float2 normal2( float seed, uint i, uint j );
}

namespace Saru
{
    __D__ inline float saru( unsigned int seed1, unsigned int seed2, unsigned int seed3 )
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

        unsigned int v = ( state ^ ( state >> 26 ) ) + wstate;
        unsigned int r = ( v ^ ( v >> 20 ) ) * 0x6957f5a7;

        float res = r / ( 4294967295.0f );
        return res;
    }

    inline __D__ float2 normal2( float seed, uint i, uint j )
    {
        float u1 = uniform01( seed, min(i, j),   max(i, j) );
        float u2 = uniform01( u1,   max(i, j)+1, min(i, j) );

        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * M_PI * u2;

        float2 res;
        sincosf(theta, &res.x, &res.y);
        res *= r;

        return res;
    }

    inline __D__ float uniform01( float seed, uint i, uint j )
    {
        float t = seed;
        unsigned int tag = *( int * )&t;

        return saru( tag, i, j );
    }

    inline __D__ float mean0var1( float seed, uint i, uint j )
    {
        return uniform01(seed, i, j) * 3.464101615f - 1.732050807f;
    }

    inline __D__ float mean0var1( float seed, int i, int j )
    {
        return mean0var1( seed, (uint) i, (uint) j );
    }

    inline __D__ float mean0var1( float seed, float i, float j )
    {
        return mean0var1( seed, (uint) i, (uint) j );
    }

    struct mean0var1_flops_counter {
        const static unsigned long long FLOPS = 2ULL;
    };

#endif
}
