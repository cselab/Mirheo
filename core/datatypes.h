#pragma once

#include <cuda.h>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include <utility>
#include <stack>
#include <algorithm>

#include "logger.h"

//==================================================================================================================
// Basic types
//==================================================================================================================

struct __align__(16) Float3_int
{
	float3 v;
	union
	{
		int32_t i;
		struct { int16_t s1, s2; };
	};

	__host__ __device__ inline Float3_int() {};
	__host__ __device__ inline Float3_int(const float3 v, int i) : v(v), i(i) {};

	__host__ __device__ inline Float3_int(const float4 f4)
	{
		v = make_float3(f4.x, f4.y, f4.z);

#ifdef __CUDA_ARCH__
		i = __float_as_int(f4.w);
#else
		union {int i; float f;} u;
		u.f = f4.w;
		i = u.i;
#endif
	}


	__host__ __device__ inline float4 toFloat4()
	{
		float f;

#ifdef __CUDA_ARCH__
		f = __int_as_float(i);
#else
		union {int i; float f;} u;
		u.i = i;
		f = u.f;
#endif

		return make_float4(v.x, v.y, v.z, f);
	}
};


struct __align__(16) Particle
{
	// We're targeting little-endian systems here, note that!

	// Free particles will have their id in i1 (or in s21*2^32 + i1)
	// Object particles will have their id (in object) in s21 and object id in i1
	// s22 is arbitrary

	float3 r;
	union
	{
		int32_t i1;
		struct { int16_t s11 /*least significant*/, s12; };
	};

	float3 u;
	union
	{
		int32_t i2;
		struct { int16_t s21 /*least significant*/, s22; };
	};

	__host__ __device__ inline Particle() {};
	__host__ __device__ inline Particle(const float4 r4, const float4 u4)
	{
		Float3_int rtmp(r4), utmp(u4);
		r  = rtmp.v;
		i1 = rtmp.i;
		u  = utmp.v;
		i2 = utmp.i;
	}
	__host__ __device__ inline Particle(const float4* coosvels, int pid)
	{
		Float3_int rtmp(coosvels[2*pid]), utmp(coosvels[2*pid+1]);
		r  = rtmp.v;
		i1 = rtmp.i;
		u  = utmp.v;
		i2 = utmp.i;
	}

	__host__ __device__ inline void readCoordinate(const float4* addr, const int pid)
	{
		const Float3_int tmp = addr[2*pid];
		r  = tmp.v;
		i1 = tmp.i;
	}

	__host__ __device__ inline void readVelocity(const float4* addr, const int pid)
	{
		const Float3_int tmp = addr[2*pid+1];
		u  = tmp.v;
		i2 = tmp.i;
	}

	__host__ __device__ inline float4 r2Float4()
	{
		return Float3_int{r, i1}.toFloat4();
	}

	__host__ __device__ inline float4 u2Float4()
	{
		return Float3_int{u, i2}.toFloat4();
	}
};


struct __align__(16) Force
{
	float3 f;
	int32_t i;

	__host__ __device__ inline Force() {};
	__host__ __device__ inline Force(const float3 f, int i) : f(f), i(i) {};

	__host__ __device__ inline Force(const float4 f4)
	{
		Float3_int tmp(f4);
		f = tmp.v;
		i = tmp.i;
	}

	__host__ __device__ inline float4 toFloat4()
	{
		return Float3_int{f, i}.toFloat4();
	}
};



