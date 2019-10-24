/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.q
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#pragma once

#include "cpu_gpu_defines.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#include <cmath>
#include <cstdlib>
#endif

using uint =  unsigned int;
using ushort =  unsigned short;

namespace math
{
static inline __HD__ float  abs(float x)  {return ::fabsf(x);}
static inline __HD__ double abs(double x) {return ::fabs (x);}
static inline __HD__ float  abs(int x)  {return ::abs(x);}
static inline __HD__ double abs(long x) {return ::abs(x);}

static inline __HD__ float  sqrt(float x)  {return ::sqrtf(x);}
static inline __HD__ double sqrt(double x) {return ::sqrt (x);}

static inline __HD__ float  ceil(float x)  {return ::ceilf(x);}
static inline __HD__ double ceil(double x) {return ::ceil (x);}

static inline __HD__ float  floor(float x)  {return ::floorf(x);}
static inline __HD__ double floor(double x) {return ::floor (x);}

static inline __HD__ float  exp(float x)  {return ::expf(x);}
static inline __HD__ double exp(double x) {return ::exp (x);}

static inline __HD__ float  cos(float x)  {return ::cosf(x);}
static inline __HD__ double cos(double x) {return ::cos (x);}

static inline __HD__ float  sin(float x)  {return ::sinf(x);}
static inline __HD__ double sin(double x) {return ::sin (x);}

static inline __HD__ float  pow(float  x, float  y) {return ::powf(x, y);}
static inline __HD__ double pow(double x, double y) {return ::pow (x, y);}

static inline __HD__ float  atan2(float  x, float  y) {return ::atan2f(x, y);}
static inline __HD__ double atan2(double x, double y) {return ::atan2 (x, y);}

#if defined(__CUDACC__)

static inline __HD__ float  rsqrt(float x)  {return ::rsqrtf(x);}
static inline __HD__ double rsqrt(double x) {return ::rsqrt (x);}

static inline __HD__ float  min(float  a, float  b) {return ::fminf(a,b);}
static inline __HD__ double min(double a, double b) {return ::min(a,b);}
static inline __HD__ int    min(int    a, int    b) {return ::min(a,b);}
static inline __HD__ uint   min(uint   a, uint   b) {return ::min(a,b);}

static inline __HD__ float  max(float  a, float  b) {return ::fmaxf(a,b);}
static inline __HD__ double max(double a, double b) {return ::max(a,b);}
static inline __HD__ int    max(int    a, int    b) {return ::max(a,b);}
static inline __HD__ uint   max(uint   a, uint   b) {return ::max(a,b);}

static inline __HD__ float2 sincos(float x)
{
    float2 res;
    ::sincosf(x, &res.x, &res.y);
    return res;
}

static inline __HD__ double2 sincos(double x)
{
    double2 res;
    ::sincos(x, &res.x, &res.y);
    return res;
}

#else

static inline float  rsqrt(float x)  {return 1.f / math::sqrt(x);}
static inline double rsqrt(double x) {return 1.0 / math::sqrt(x);}

template <typename T> static inline T min(const T& a, const T& b) {return a < b ? a : b;}
template <typename T> static inline T max(const T& a, const T& b) {return a < b ? b : a;}

static inline __HD__ float2  sincos(float x)  {return {sin(x), cos(x)};}
static inline __HD__ double2 sincos(double x) {return {sin(x), cos(x)};}

#endif

} // namespace math



////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
static inline __HD__ float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
static inline __HD__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
static inline __HD__ float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

static inline __HD__ int2 make_int2(int s)
{
    return make_int2(s, s);
}
static inline __HD__ int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
static inline __HD__ int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
static inline __HD__ int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

static inline __HD__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
static inline __HD__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
static inline __HD__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
static inline __HD__ float3 make_float3(float3 a)
{
    return a;
}
static inline __HD__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
static inline __HD__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
static inline __HD__ float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

static inline __HD__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
static inline __HD__ int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
static inline __HD__ int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
static inline __HD__ int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
static inline __HD__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

static inline __HD__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
static inline __HD__ uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
static inline __HD__ uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
static inline __HD__ uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
static inline __HD__ uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

static inline __HD__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
static inline __HD__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
static inline __HD__ float4 make_float4(float x, float3 a)
{
    return make_float4(x, a.x, a.y, a.z);
}
static inline __HD__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
static inline __HD__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
static inline __HD__ float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

static inline __HD__ int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
static inline __HD__ int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
static inline __HD__ int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
static inline __HD__ int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
static inline __HD__ int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


static inline __HD__ uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
static inline __HD__ uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
static inline __HD__ uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
static inline __HD__ uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}


static inline __HD__ float3 make_float3(double3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
static inline __HD__ double3 make_double3(float3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}

static inline __HD__ float4 make_float4(double4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
static inline __HD__ double4 make_double4(float4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 operator-(const float2 &a)
{
    return make_float2(-a.x, -a.y);
}
static inline __HD__ int2 operator-(const int2 &a)
{
    return make_int2(-a.x, -a.y);
}
static inline __HD__ float3 operator-(const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
static inline __HD__ double3 operator-(const double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}
static inline __HD__ int3 operator-(const int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
static inline __HD__ float4 operator-(const float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
static inline __HD__ double4 operator-(const double4 &a)
{
    return make_double4(-a.x, -a.y, -a.z, -a.w);
}
static inline __HD__ int4 operator-(const int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
static inline __HD__ double2 operator+(double2 a, double2 b)
{
    return {a.x + b.x, a.y + b.y};
}
static inline __HD__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
static inline __HD__ void operator+=(double2 &a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}
static inline __HD__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
static inline __HD__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
static inline __HD__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

static inline __HD__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
static inline __HD__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
static inline __HD__ int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
static inline __HD__ int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
static inline __HD__ void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

static inline __HD__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
static inline __HD__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
static inline __HD__ uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
static inline __HD__ uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
static inline __HD__ void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


static inline __HD__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline __HD__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
static inline __HD__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
static inline __HD__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

static inline __HD__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline __HD__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
static inline __HD__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
static inline __HD__ void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

static inline __HD__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline __HD__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
static inline __HD__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
static inline __HD__ void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

static inline __HD__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
static inline __HD__ uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
static inline __HD__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

static inline __HD__ double3 operator+(double b, double3 a)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}

static inline __HD__ double3 operator+(double3 b, double a)
{
    return make_double3(a + b.x, a + b.y, a + b.z);
}

static inline __HD__ double3 operator+(double3 b, double3 a)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline __HD__ double3 operator+(double3 b, float3 a)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline __HD__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
static inline __HD__ void operator+=(double3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}


static inline __HD__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
static inline __HD__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
static inline __HD__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
static inline __HD__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
static inline __HD__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

static inline __HD__ double4 operator+(double4 a, double4 b)
{
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
static inline __HD__ void operator+=(double4 &a, double4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
static inline __HD__ double4 operator+(double4 a, double b)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
static inline __HD__ double4 operator+(double b, double4 a)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
static inline __HD__ void operator+=(double4 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

static inline __HD__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
static inline __HD__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
static inline __HD__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
static inline __HD__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
static inline __HD__ void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

static inline __HD__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
static inline __HD__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
static inline __HD__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
static inline __HD__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
static inline __HD__ void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
static inline __HD__ double2 operator-(double2 a, double2 b)
{
    return {a.x - b.x, a.y - b.y};
}
static inline __HD__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
static inline __HD__ void operator-=(double2 &a, double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
static inline __HD__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
static inline __HD__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
static inline __HD__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

static inline __HD__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
static inline __HD__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
static inline __HD__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
static inline __HD__ int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
static inline __HD__ void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

static inline __HD__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
static inline __HD__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
static inline __HD__ uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
static inline __HD__ uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
static inline __HD__ void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

static inline __HD__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static inline __HD__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
static inline __HD__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
static inline __HD__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
static inline __HD__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

static inline __HD__ double3 operator-(double a, double3 b)
{
    return make_double3(a - b.x, a - b.y, a - b.z);
}

static inline __HD__ double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}

static inline __HD__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static inline __HD__ void operator-=(double3 &a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

static inline __HD__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static inline __HD__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
static inline __HD__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
static inline __HD__ int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
static inline __HD__ void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

static inline __HD__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static inline __HD__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
static inline __HD__ uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
static inline __HD__ uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
static inline __HD__ void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

static inline __HD__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
static inline __HD__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
static inline __HD__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
static inline __HD__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

static inline __HD__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
static inline __HD__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
static inline __HD__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
static inline __HD__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
static inline __HD__ void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

static inline __HD__ double4 operator-(double4 a, double4 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w};
}
static inline __HD__ void operator-=(double4 &a, double4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
static inline __HD__ double4 operator-(double4 a, double b)
{
    return {a.x - b, a.y - b, a.z - b,  a.w - b};
}
static inline __HD__ void operator-=(double4 &a, double b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

static inline __HD__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
static inline __HD__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
static inline __HD__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
static inline __HD__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
static inline __HD__ void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
static inline __HD__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
static inline __HD__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
static inline __HD__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
static inline __HD__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

static inline __HD__ double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}
static inline __HD__ void operator*=(double2 &a, double2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
static inline __HD__ double2 operator*(double2 a, double b)
{
    return make_double2(a.x * b, a.y * b);
}
static inline __HD__ double2 operator*(double b, double2 a)
{
    return make_double2(b * a.x, b * a.y);
}
static inline __HD__ void operator*=(double2 &a, double b)
{
    a.x *= b;
    a.y *= b;
}

static inline __HD__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
static inline __HD__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
static inline __HD__ int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
static inline __HD__ int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
static inline __HD__ void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

static inline __HD__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
static inline __HD__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
static inline __HD__ uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
static inline __HD__ uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
static inline __HD__ void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

static inline __HD__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static inline __HD__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
static inline __HD__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
static inline __HD__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
static inline __HD__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

static inline __HD__ double3 operator*(float3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static inline __HD__ double3 operator*(double3 a, float3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static inline __HD__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static inline __HD__ void operator*=(double3 &a, double3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
static inline __HD__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
static inline __HD__ double3 operator*(double b, double3 a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}
static inline __HD__ void operator*=(double3 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}


static inline __HD__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static inline __HD__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
static inline __HD__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
static inline __HD__ int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
static inline __HD__ void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

static inline __HD__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static inline __HD__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
static inline __HD__ uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
static inline __HD__ uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
static inline __HD__ void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

static inline __HD__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
static inline __HD__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
static inline __HD__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
static inline __HD__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
static inline __HD__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}


static inline __HD__ double4 operator*(double4 a, double4 b)
{
    return make_double4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
static inline __HD__ void operator*=(double4 &a, double4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
static inline __HD__ double4 operator*(double4 a, double b)
{
    return make_double4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
static inline __HD__ double4 operator*(double b, double4 a)
{
    return make_double4(b * a.x, b * a.y, b * a.z, b * a.w);
}
static inline __HD__ void operator*=(double4 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}


static inline __HD__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
static inline __HD__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
static inline __HD__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
static inline __HD__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
static inline __HD__ void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

static inline __HD__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
static inline __HD__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
static inline __HD__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
static inline __HD__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
static inline __HD__ void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
static inline __HD__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
static inline __HD__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
static inline __HD__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
static inline __HD__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

static inline __HD__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
static inline __HD__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
static inline __HD__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
static inline __HD__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
static inline __HD__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

static inline __HD__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
static inline __HD__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
static inline __HD__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
static inline __HD__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
static inline __HD__ float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}



static inline __HD__ double2 operator/(double2 a, double2 b)
{
    return make_double2(a.x / b.x, a.y / b.y);
}
static inline __HD__ void operator/=(double2 &a, double2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
static inline __HD__ double2 operator/(double2 a, double b)
{
    return make_double2(a.x / b, a.y / b);
}
static inline __HD__ void operator/=(double2 &a, double b)
{
    a.x /= b;
    a.y /= b;
}
static inline __HD__ double2 operator/(double b, double2 a)
{
    return make_double2(b / a.x, b / a.y);
}

static inline __HD__ double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
static inline __HD__ void operator/=(double3 &a, double3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
static inline __HD__ double3 operator/(double3 a, double b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}
static inline __HD__ void operator/=(double3 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
static inline __HD__ double3 operator/(double b, double3 a)
{
    return make_double3(b / a.x, b / a.y, b / a.z);
}

static inline __HD__ double4 operator/(double4 a, double4 b)
{
    return make_double4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
static inline __HD__ void operator/=(double4 &a, double4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
static inline __HD__ double4 operator/(double4 a, double b)
{
    return make_double4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
static inline __HD__ void operator/=(double4 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
static inline __HD__ double4 operator/(double b, double4 a)
{
    return make_double4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// Extra division-related stuff
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ int3 operator%(int3 a, int3 b)
{
    return make_int3(a.x % b.x, a.y % b.y, a.z % b.z);
}

static inline __HD__ int3 operator/(int3 a, int b)
{
    return make_int3(a.x / b, a.y / b, a.z / b);
}

static inline __HD__ int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}


////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
static inline __HD__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline __HD__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static inline __HD__ double dot(double2 a, double2 b)
{
    return a.x * b.x + a.y * b.y;
}
static inline __HD__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline __HD__ double dot(double4 a, double4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static inline __HD__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
static inline __HD__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline __HD__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}


////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float length(float2 v)
{
    return math::sqrt(dot(v, v));
}
static inline __HD__ float length(float3 v)
{
    return math::sqrt(dot(v, v));
}
static inline __HD__ float length(float4 v)
{
    return math::sqrt(dot(v, v));
}

static inline __HD__ double length(double2 v)
{
    return math::sqrt(dot(v, v));
}
static inline __HD__ double length(double3 v)
{
    return math::sqrt(dot(v, v));
}
static inline __HD__ double length(double4 v)
{
    return math::sqrt(dot(v, v));
}



static inline __HD__ float distance2(float3 a, float3 b)
{
    auto sqr = [] (float x) { return x*x; };
    return sqr(a.x - b.x) + sqr(a.y - b.y) + sqr(a.z - b.z);
}

static inline __HD__ double distance2(double3 a, double3 b)
{
    auto sqr = [] (double x) { return x*x; };
    return sqr(a.x - b.x) + sqr(a.y - b.y) + sqr(a.z - b.z);
}


////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 normalize(float2 v)
{
    float invLen = math::rsqrt(dot(v, v));
    return v * invLen;
}
static inline __HD__ float3 normalize(float3 v)
{
    float invLen = math::rsqrt(dot(v, v));
    return v * invLen;
}
static inline __HD__ float4 normalize(float4 v)
{
    float invLen = math::rsqrt(dot(v, v));
    return v * invLen;
}


static inline __HD__ double2 normalize(double2 v)
{
    double invLen = math::rsqrt(dot(v, v));
    return v * invLen;
}
static inline __HD__ double3 normalize(double3 v)
{
    double invLen = math::rsqrt(dot(v, v));
    return v * invLen;
}
static inline __HD__ double4 normalize(double4 v)
{
    double invLen = math::rsqrt(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

static inline __HD__ double3 cross(double3 a, double3 b)
{
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}


////////////////////////////////////////////////////////////////////////////////
// anyOrthogonal
// returns any orthogonal vector to the input vector
////////////////////////////////////////////////////////////////////////////////

template <class R3>
static inline  __HD__ R3 anyOrthogonal(R3 v)
{
    const auto x = fabsf(v.x);
    const auto y = fabsf(v.y);
    const auto z = fabsf(v.z);

    constexpr R3 xAxis {1.f, 0.f, 0.f};
    constexpr R3 yAxis {0.f, 1.f, 0.f};
    constexpr R3 zAxis {0.f, 0.f, 1.f};
    
    auto other = x < y ? (x < z ? xAxis : zAxis) : (y < z ? yAxis : zAxis);
    return cross(v, other);
}


namespace math
{

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

static inline  __HD__ float2 min(float2 a, float2 b)
{
    return make_float2(min(a.x, b.x), min(a.y,b.y));
}
static inline __HD__ float3 min(float3 a, float3 b)
{
    return make_float3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
static inline  __HD__ float4 min(float4 a, float4 b)
{
    return make_float4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

static inline __HD__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}
static inline __HD__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
static inline __HD__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 max(float2 a, float2 b)
{
    return make_float2(max(a.x,b.x), max(a.y,b.y));
}
static inline __HD__ float3 max(float3 a, float3 b)
{
    return make_float3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
static inline __HD__ float4 max(float4 a, float4 b)
{
    return make_float4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

static inline __HD__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}
static inline __HD__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
static inline __HD__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}


////////////////////////////////////////////////////////////////////////////////
// ceil
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 ceil(float2 v)
{
    return make_float2(math::ceil(v.x), math::ceil(v.y));
}
static inline __HD__ float3 ceil(float3 v)
{
    return make_float3(math::ceil(v.x), math::ceil(v.y), math::ceil(v.z));
}
static inline __HD__ float4 ceil(float4 v)
{
    return make_float4(math::ceil(v.x), math::ceil(v.y), math::ceil(v.z), math::ceil(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 floor(float2 v)
{
    return make_float2(math::floor(v.x), math::floor(v.y));
}
static inline __HD__ float3 floor(float3 v)
{
    return make_float3(math::floor(v.x), math::floor(v.y), math::floor(v.z));
}
static inline __HD__ float4 floor(float4 v)
{
    return make_float4(math::floor(v.x), math::floor(v.y), math::floor(v.z), math::floor(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

static inline __HD__ float2 abs(float2 v)
{
    return make_float2(math::abs(v.x), math::abs(v.y));
}
static inline __HD__ float3 abs(float3 v)
{
    return make_float3(math::abs(v.x), math::abs(v.y), math::abs(v.z));
}
static inline __HD__ float4 abs(float4 v)
{
    return make_float4(math::abs(v.x), math::abs(v.y), math::abs(v.z), math::abs(v.w));
}


static inline __HD__ double2 abs(double2 v)
{
    return make_double2(math::abs(v.x), math::abs(v.y));
}
static inline __HD__ double3 abs(double3 v)
{
    return make_double3(math::abs(v.x), math::abs(v.y), math::abs(v.z));
}
static inline __HD__ double4 abs(double4 v)
{
    return make_double4(math::abs(v.x), math::abs(v.y), math::abs(v.z), math::abs(v.w));
}


static inline __HD__ int2 abs(int2 v)
{
    return make_int2(math::abs(v.x), math::abs(v.y));
}
static inline __HD__ int3 abs(int3 v)
{
    return make_int3(math::abs(v.x), math::abs(v.y), math::abs(v.z));
}
static inline __HD__ int4 abs(int4 v)
{
    return make_int4(math::abs(v.x), math::abs(v.y), math::abs(v.z), math::abs(v.w));
}

} // namespace math

