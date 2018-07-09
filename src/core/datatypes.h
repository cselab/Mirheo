#pragma once

#include <cuda.h>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include <utility>
#include <stack>
#include <algorithm>

#include <core/logger.h>

#include <cuda_runtime.h>

#ifdef __NVCC__ 
#define CUDAHOSTDEVICE __host__ __device__
#else
#define CUDAHOSTDEVICE
#endif

//==================================================================================================================
// Basic types
//==================================================================================================================

/**
 * Helper class for packing/unpacking \e float3 + \e int into \e float4
 */
struct __align__(16) Float3_int
{
    float3 v;
    int32_t i;


    CUDAHOSTDEVICE inline Float3_int(const Float3_int& x)
    {
        *((float4*)this) = *((float4*)&x);
    }

    CUDAHOSTDEVICE inline Float3_int& operator=(Float3_int x)
    {
        *((float4*)this) = *((float4*)&x);
        return *this;
    }

    CUDAHOSTDEVICE inline Float3_int() {};
    CUDAHOSTDEVICE inline Float3_int(const float3 v, int i) : v(v), i(i) {};

    CUDAHOSTDEVICE inline Float3_int(const float4 f4)
    {
        *((float4*)this) = f4;
    }

    CUDAHOSTDEVICE inline float4 toFloat4() const
    {
        float4 f = *((float4*)this);
        return f;
    }
};

/**
 * Structure to hold coordinates and velocities of particles.
 * Due to performance reasons it should be aligned to 16 bytes boundary,
 * therefore 8 bytes = 2 integer numbers are extra.
 * The integer fields are used for particle ids
 *
 * For performance reasons instead of an N-element array of Particle
 * an array of 2*N \e float4 elements is used.
 */
struct __align__(16) Particle
{
    float3 r;    ///< coordinate
    int32_t i1;  ///< lower part of particle id

    float3 u;    ///< velocity
    int32_t i2;  ///< higher part of particle id

    /// Copy constructor uses efficient 16-bytes wide copies
    CUDAHOSTDEVICE inline Particle(const Particle& x)
    {
        auto f4this = (float4*)this;
        auto f4x    = (float4*)&x;

        f4this[0] = f4x[0];
        f4this[1] = f4x[1];
    }

    /// Assignment operator uses efficient 16-bytes wide copies
    CUDAHOSTDEVICE inline Particle& operator=(Particle x)
    {
        auto f4this = (float4*)this;
        auto f4x    = (float4*)&x;

        f4this[0] = f4x[0];
        f4this[1] = f4x[1];

        return *this;
    }

    /**
     * Default constructor
     *
     * @rst
     * .. attention::
     *    Default constructor DOES NOT set any members!
     * @endrst
     */
    CUDAHOSTDEVICE inline Particle() {};

    /**
     * Construct a Particle from two float4 entries
     *
     * @param r4 first three floats will be coordinates (#r), last one, \e \.w - #i1
     * @param u4 first three floats will be velocities (#u), last one \e \.w - #i1
     */
    CUDAHOSTDEVICE inline Particle(const float4 r4, const float4 u4)
    {
        Float3_int rtmp(r4), utmp(u4);
        r  = rtmp.v;
        i1 = rtmp.i;
        u  = utmp.v;
        i2 = utmp.i;
    }

    /**
     * Equivalent to the following constructor:
     * @code
     * Particle(addr[2*pid], addr[2*pid+1]);
     * @endcode
     *
     * @param addr must have at least \e 2*pid entries
     * @param pid  particle id
     */
    CUDAHOSTDEVICE inline Particle(const float4* addr, int pid)
    {
        readCoordinate(addr, pid);
        readVelocity  (addr, pid);
    }


    /**
     * Only read coordinates from the given \e addr
     *
     * @param addr must have at least \e 2*pid entries
     * @param pid  particle id
     */
    CUDAHOSTDEVICE inline void readCoordinate(const float4* addr, const int pid)
    {
        const Float3_int tmp = addr[2*pid];
        r  = tmp.v;
        i1 = tmp.i;
    }

    /**
     * Only read velocities from the given \e addr
     *
     * @param addr must have at least \e 2*pid entries
     * @param pid  particle id
     */
    CUDAHOSTDEVICE inline void readVelocity(const float4* addr, const int pid)
    {
        const Float3_int tmp = addr[2*pid+1];
        u  = tmp.v;
        i2 = tmp.i;
    }

    /**
     * Helps writing particles back to \e float4 array
     *
     * @return packed #r and #i1 as \e float4
     */
    CUDAHOSTDEVICE inline float4 r2Float4() const
    {
        return Float3_int{r, i1}.toFloat4();
    }

    /**
     * Helps writing particles back to \e float4 array
     *
     * @return packed #u and #i2 as \e float4
     */
    CUDAHOSTDEVICE inline float4 u2Float4() const
    {
        return Float3_int{u, i2}.toFloat4();
    }

    /**
     * Helps writing particles back to \e float4 array
     *
     * @param dst must have at least \e 2*pid entries
     * @param pid particle id
     */
    CUDAHOSTDEVICE inline void write2Float4(float4* dst, int pid) const
    {
        dst[2*pid]   = r2Float4();
        dst[2*pid+1] = u2Float4();
    }
};

/**
 * Structure holding force
 * Not much used as of now
 */
struct __align__(16) Force
{
    float3 f;
    int32_t i;

    CUDAHOSTDEVICE inline Force() {};
    CUDAHOSTDEVICE inline Force(const float3 f, int i) : f(f), i(i) {};

    CUDAHOSTDEVICE inline Force(const float4 f4)
    {
        Float3_int tmp(f4);
        f = tmp.v;
        i = tmp.i;
    }

    CUDAHOSTDEVICE inline float4 toFloat4()
    {
        return Float3_int{f, i}.toFloat4();
    }
};



