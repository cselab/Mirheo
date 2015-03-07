#pragma once

namespace Logistic
{
    __device__ float mean0var1(float seed, uint i, uint j );	
}

#include <cstdlib>
#include <limits>
#include <stdint.h>
#include <cmath>

namespace Logistic
{
/************************* Trunk generator ***********************
 * Make one global random number per each timestep
 * cite G. Marsaglia
 * passes BigCrush
 *****************************************************************/
    struct KISS 
    {
	typedef uint32_t integer;
	integer x, y, z, c;

    KISS() : x(0), y(0), z(0), c(0) {}

    KISS( integer x_, integer y_, integer z_, integer c_ ) :
	x(x_), y(y_), z(z_), c(c_) {}
	
	float get_float() 
	    {
		return get_int() / float(std::numeric_limits<integer>::max());
	    }

	integer get_int() 
	    {
		uint64_t t, a = 698769069ULL;
		x = 69069*x+12345;
		y ^= (y<<13); y ^= (y>>17); y ^= (y<<5); /* y must never be set to zero! */
		t = a*z+c; c = (t>>32); /* Also avoid setting z=c=0! */
		return x+y+(z=t);
	    }
    };

#ifdef __CUDACC__

/************************* Branch generator **********************
 * Make one random number per pair of particles per timestep
* Based on the Logistic map on interval [-1,1]
* each branch no weaker than trunk
* zero dependency between branches
*****************************************************************/
// passes of logistic map
    const static int N = 18;
// spacing coefficints for low discrepancy numbers
    const static float gold = 0.6180339887498948482;
    const static float silver = 0.4142135623730950488;
    const static float bronze = 0.00008877875787352212838853023;
    const static float tin = 0.00004602357186447026756768986;
// square root of 2
    const static float sqrt2 = 1.41421356237309514547;

// floating point version of LCG
    __inline__ __device__ float rem( float r ) 
    {
	return r - floor(r);
    }

// FMA wrapper for the convenience of switching rouding modes
    __inline__ __device__ float FMA( float x, float y, float z ) 
    {
	return __fmaf_rz(x, y, z);
    }

// logistic rounds
// <3> : 4 FMA + 1 MUL
// <2> : 2 FMA + 1 MUL
// <1> : 1 FMA + 1 MUL
    template<int N> __inline__ __device__ float __logistic_core( float x ) 
    {
	float x2 = x * x;
	float r = FMA( FMA( 8.0, x2, -8.0 ), x2, 1.0 );
	return __logistic_core<N-2>( r );
    }

    template<> __inline__ __device__ float __logistic_core<1>( float x ) 
    {
	return FMA( 2.0 * x, x, -1.0 );
    }

    template<> __inline__ __device__ float __logistic_core<0>( float x ) 
    {
	return x;
    }

// random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
// mean = 0
// variance = 1
// can be used directly for DPD
#if 1
    __inline__ __device__ float mean0var1( float seed, uint i, uint j )
    {
	uint u = min( i, j );
	uint v = max( i, j );
	//float p = rem( u * gold + v * silver ); // unsafe for large u or v
	float p = rem( ( (u&0x3FF)*gold) + u*bronze + ((v&0x3FF)*silver) + v * tin ); // safe for large u or v
	float l = __logistic_core<N>( seed - p );
	return l * sqrt2;
    }
#else
//    __device__ inline float saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
//    {
//	seed3 ^= (seed1<<7)^(seed2>>6);
//	seed2 += (seed1>>4)^(seed3>>15);
//	seed1 ^= (seed2<<9)+(seed3<<8);
//	seed3 ^= 0xA5366B4D*((seed2>>11) ^ (seed1<<1));
//	seed2 += 0x72BE1579*((seed1<<4)  ^ (seed3>>16));
//	seed1 ^= 0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
//	seed2 += seed1*seed3;
//	seed1 += seed3 ^ (seed2>>2);
//	seed2 ^= ((signed int)seed2)>>17;
//
//	int state  = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
//	int wstate = (state + seed2) ^ (((signed int)state)>>8);
//	state  = state + (wstate*(wstate^0xdddf97f5));
//	wstate = 0xABCB96F7 + (wstate>>1);
//
//	state  = 0x4beb5d59*state + 0x2600e1f7; // LCG
//	wstate = wstate + 0x8009d14b + ((((signed int)wstate)>>31)&0xda879add); // OWS
//
//	unsigned int v = (state ^ (state>>26))+wstate;
//	unsigned int r = (v^(v>>20))*0x6957f5a7;
//
//	float res = r / (4294967295.0f);
//	return res;
//    }
//
//    __inline__ __device__ float mean0var1( float seed, uint i, uint j )
//    {
//	float t = seed;
//	unsigned int tag = *(int *)&t;
//
//	return saru(tag, i, j) * 3.464101615f - 1.732050807f;
//    }
#endif
#endif
}
