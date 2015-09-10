/*
 *  logistic.h
 *  Part of uDeviceX/logistic_rng/
 *
 *  Created and authored by Yu-Hang Tang on 2015-03-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#ifndef __LOGISTIC_RNG__
#define __LOGISTIC_RNG__

#include<cuda.h>

namespace logi {
__inline__ __device__ double floor( double x ) { return ::floor(x); }
__inline__ __device__ float  floor( float  x ) { return ::floorf(x); }
__inline__ __device__ double fma( double x, double y, double z ) { return ::fma(x,y,z); }
__inline__ __device__ float  fma( float  x, float  y, float z  ) { return ::fmaf(x,y,z); }
}

template<int N> __inline__ __device__ double logistic_core( double x ) {
	x = 4.0 * fma( x, -x, x );
	return logistic_core<N-1>(x);
}
template<> __inline__ __device__ double logistic_core<0>( double x ) {
	return x;
}

template<int N> __inline__ __device__ float logistic_core( float x ) {
	x = 4.0f * fmaf( x, -x, x );
	return logistic_core<N-1>(x);
}
template<> __inline__ __device__ float logistic_core<0>( float x ) {
	return x;
}

// (sqrt(5)-1)/2
#define __GOLD__   0.6180339887498948482
// sqrt(2)-1
#define __SILVER__ 0.4142135623730950488

template<int N, typename REAL> __inline__ __device__
REAL logistic( REAL trunk, int i, int j ) {
	uint u = min(i,j);
	uint v = max(i,j);
	REAL p = logi::fma( u, REAL(__GOLD__),   -logi::floor( u * REAL(__GOLD__)   ) );
	REAL q = logi::fma( v, REAL(__SILVER__), -logi::floor( v * REAL(__SILVER__) ) );
    REAL r = trunk * ( p + q - logi::floor(p+q) );
    return logistic_core<N>( r );
}

#endif


