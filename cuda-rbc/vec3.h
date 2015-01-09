//
//  vec3.h
//  vanilla-rbc
//
//  Created by Dmitry Alexeev on 06/11/14.
//  Copyright (c) 2014 ETH Zurich. All rights reserved.
//

#include <cmath>
#include "misc.h"


#pragma once

#ifndef __host__
#define __host__
#define __device__
#endif

#ifndef nullptr
#define nullptr 0
#endif

struct vec3
{
    real x, y, z;
    
    __host__ __device__ vec3(real x0, real x1, real x2)
    {
        x = x0;
        y = x1;
        z = x2;
    }
    
    __host__ __device__ vec3(const real *r)
    {
        x = r[0];
        y = r[1];
        z = r[2];
    }
    
    __host__ __device__ vec3() : x(0), y(0), z(0) { };
    
    __host__ __device__ vec3& operator=(const vec3& u)
    {
        if (&u == this) return *this;
        x = u.x;
        y = u.y;
        z = u.z;
        return *this;
    }
    
    __host__ __device__ vec3  operator+(const vec3& u) const
    {
        vec3 res(x+u.x, y+u.y, z+u.z);
        return res;
    }
    
    __host__ __device__ vec3  operator-(const vec3& u) const
    {
        vec3 res(x-u.x, y-u.y, z-u.z);
        return res;
    }
    
    __host__ __device__ vec3  operator/(const real& a) const
    {
        vec3 res(x/a, y/a, z/a);
        return res;
    }
    
    __host__ __device__ vec3  operator*(const real& a) const
    {
        vec3 res(x*a, y*a, z*a);
        return res;
    }
    
    __host__ __device__ vec3& operator+=(const vec3& u)
    {
        x += u.x;
        y += u.y;
        z += u.z;
        return *this;
    }
    
    __host__ __device__ vec3& operator-=(const vec3& u)
    {
        x -= u.x;
        y -= u.y;
        z -= u.z;
        return *this;
    }
    
    __host__ __device__ vec3& operator/=(const real& a)
    {
        x /= a;
        y /= a;
        z /= a;
        return *this;
    }
    
    __host__ __device__ vec3& operator*=(const real& a)
    {
        x *= a;
        y *= a;
        z *= a;
        return *this;
    }
    
    __host__ __device__ vec3  operator-() const
    {
        vec3 res(-x, -y, -z);
        return res;
    }
    
    __host__ __device__ real& operator[](int i)
    {
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;

	//abort();
        return x;
    }
};

__host__ __device__ inline real dot(const vec3 u, const vec3 v)
{
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

__host__ __device__ inline real norm(const vec3 u)
{
    return sqrt(dot(u, u));
}

__host__ __device__ inline void normalize(vec3& u)
{
    real n = norm(u);
    if (n < 1e-10)
    {
        u.x = u.y = u.z = 0;
    }
    else
    {
        u.x /= n;
        u.y /= n;
        u.z /= n;
    }
}

__host__ __device__ inline void normalize(const vec3 u, vec3& res)
{
    real n = norm(u);
    if (n < 1e-10)
    {
        res.x = res.y = res.z = 0;
    }
    else
    {
        res.x = u.x/n;
        res.y = u.y/n;
        res.z = u.z/n;
    }
}

__host__ __device__ inline real _dist(const vec3 x, const vec3 y)
{
    return sqrt( (x.x-y.x)*(x.x-y.x) + (x.y-y.y)*(x.y-y.y) + (x.z-y.z)*(x.z-y.z) );
}

__host__ __device__ inline real _dist2(const vec3 x, const vec3 y)
{
    return (x.x-y.x)*(x.x-y.x) + (x.y-y.y)*(x.y-y.y) + (x.z-y.z)*(x.z-y.z);
}

__host__ __device__ inline vec3 cross(const vec3 a, const vec3 b)
{
    vec3 res;
    res.x = a.y*b.z - a.z*b.y;
    res.y = a.z*b.x - a.x*b.z;
    res.z = a.x*b.y - a.y*b.x;
    return res;
}

__host__ __device__ inline vec3 orthpart(const vec3 base, const vec3 v)
{
    return v - base * dot(base, v);
}
