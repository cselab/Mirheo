/*
 *  logistic.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#ifndef _LOGISTIC_H
#define _LOGISTIC_H

#include <cstdlib>
#include <limits>
#include <stdint.h>
#include <cmath>

/************************* Trunk generator ***********************
 * Make one global random number per each timestep
 * cite G. Marsaglia
 * passes BigCrush
 *****************************************************************/
struct KISS {
	typedef uint32_t integer;

        integer x, y, z, c;
	
	KISS( integer x_, integer y_, integer z_, integer c_ ) :
		x(x_), y(y_), z(z_), c(c_) {}

        float get_float() {
                return get_int() / float(std::numeric_limits<integer>::max());
        }

        integer get_int() {
                uint64_t t, a = 698769069ULL;
                x = 69069*x+12345;
                y ^= (y<<13); y ^= (y>>17); y ^= (y<<5); /* y must never be set to zero! */
                t = a*z+c; c = (t>>32); /* Also avoid setting z=c=0! */
                return x+y+(z=t);
        }
};

/************************* Branch generator **********************
 * Make one random number per pair of particles per timestep
 * Based on the Logistic map on interval [-1,1]  
 * each branch no weak than trunk
 * zero dependency between branches
 *****************************************************************/

// passes of logistic map
const static int N = 18;
// spacing coefficints for low discrepancy numbers
const static float gold = 0.6180339887498948482;
const static float silver = 0.4142135623730950488;
// square root of 2
const static float sqrt2 = 1.41421356237309514547;

// floating point version of LCG
__inline__ __device__ float rem( float r ) {
	return r - floor(r);
}

// FMA wrapper for the convenience of switching rouding modes
__inline__ __device__ float FMA( float x, float y, float z ) {
	return __fmaf_rz(x, y, z);
}

// logistic rounds
// <3> : 4 FMA + 1 MUL
// <2> : 2 FMA + 1 MUL
// <1> : 1 FMA + 1 MUL
template<int N> __inline__ __device__ float __logistic_core( float x ) {
	float x2 = x * x;
	float r = FMA( FMA( 8.0, x2, -8.0 ), x2, 1.0 );
	return __logistic_core<N-2>( r );
}
template<> __inline__ __device__ float __logistic_core<1>( float x ) {
	return FMA( 2.0 * x, x, -1.0 );
}
template<> __inline__ __device__ float __logistic_core<0>( float x ) {
	return x;
}

// random number from the ArcSine distribution on [-sqrt(2),sqrt(2)]
// mean = 0
// variance = 1
// can be used directly for DPD
__inline__ __device__ float mean0var1( float seed, uint i, uint j ) {
    uint u = min( i, j );
    uint v = max( i, j );
    float p = rem( u * gold + v * silver );
    float l = __logistic_core<N>( seed - p );
    return l * sqrt2;
}


#endif
