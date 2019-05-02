#pragma once

#include <core/logger.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/vec_traits.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>

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

    static constexpr float mark_val = -900.f;

    __HD__ inline Float3_int(const Float3_int& x)
    {
        *((float4*)this) = *((float4*)&x);
    }

    __HD__ inline Float3_int& operator=(Float3_int x)
    {
        *((float4*)this) = *((float4*)&x);
        return *this;
    }

    __HD__ inline Float3_int() {};
    __HD__ inline Float3_int(const float3 v, int i) : v(v), i(i) {};

    __HD__ inline Float3_int(const float4 f4)
    {
        *((float4*)this) = f4;
    }

    __HD__ inline float4 toFloat4() const
    {
        float4 f = *((float4*)this);
        return f;
    }

    __HD__ inline void mark()
    {
        v.x = v.y = v.z = mark_val;
    }

    __HD__ inline bool isMarked() const
    {
        return v.x == mark_val && v.y == mark_val && v.z == mark_val;
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
    __HD__ inline Particle(const Particle& x)
    {
        auto f4this = (float4*)this;
        auto f4x    = (float4*)&x;

        f4this[0] = f4x[0];
        f4this[1] = f4x[1];
    }

    /// Assignment operator uses efficient 16-bytes wide copies
    __HD__ inline Particle& operator=(Particle x)
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
    __HD__ inline Particle() {};

    __HD__ inline void setId(int64_t id)
    {
        int64_t highHalf = (id >> 32) << 32;
        i1 = (int32_t) (id - highHalf);
        i2 = (int32_t) (id >> 32);
    }

    __HD__ inline int64_t getId() const
    {
        return int64_t(i1) + (int64_t (i2) << 32);
    }
    
    /**
     * Construct a Particle from two float4 entries
     *
     * @param r4 first three floats will be coordinates (#r), last one, \e \.w - #i1
     * @param u4 first three floats will be velocities (#u), last one \e \.w - #i1
     */
    __HD__ inline Particle(const float4 r4, const float4 u4)
    {
        Float3_int rtmp(r4), utmp(u4);
        r  = rtmp.v;
        i1 = rtmp.i;
        u  = utmp.v;
        i2 = utmp.i;
    }

    /**
     * Only read coordinates from the given \e addr
     *
     * @param addr must have at least \e pid entries
     * @param pid  particle id
     */
    __HD__ inline void readCoordinate(const float4 *addr, const int pid)
    {
        const Float3_int tmp = addr[pid];
        r  = tmp.v;
        i1 = tmp.i;
    }

    /**
     * Only read velocities from the given \e addr
     *
     * @param addr must have at least \e pid entries
     * @param pid  particle id
     */
    __HD__ inline void readVelocity(const float4 *addr, const int pid)
    {
        const Float3_int tmp = addr[pid];
        u  = tmp.v;
        i2 = tmp.i;
    }

    __HD__ inline Float3_int r2Float3_int() const
    {
        return Float3_int{r, i1};
    }
    
    /**
     * Helps writing particles back to \e float4 array
     *
     * @return packed #r and #i1 as \e float4
     */
    __HD__ inline float4 r2Float4() const
    {
        return r2Float3_int().toFloat4();
    }

    __HD__ inline Float3_int u2Float3_int() const
    {
        return Float3_int{u, i2};
    }

    /**
     * Helps writing particles back to \e float4 array
     *
     * @return packed #u and #i2 as \e float4
     */
    __HD__ inline float4 u2Float4() const
    {
        return u2Float3_int().toFloat4();
    }

    /**
     * Helps writing particles back to \e float4 array
     *
     * @param dst must have at least \e 2*pid entries
     * @param pid particle id
     */
    __HD__ inline void write2Float4(float4* pos, float4 *vel, int pid) const
    {
        pos[pid] = r2Float4();
        vel[pid] = u2Float4();
    }

    __HD__ inline void mark()
    {
        Float3_int f3i = r2Float3_int();
        f3i.mark();
        r = f3i.v;
    }

    __HD__ inline bool isMarked() const
    {
        return r2Float3_int().isMarked();
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

    __HD__ inline Force() {};
    __HD__ inline Force(const float3 f, int i) : f(f), i(i) {};

    __HD__ inline Force(const float4 f4)
    {
        Float3_int tmp(f4);
        f = tmp.v;
        i = tmp.i;
    }

    __HD__ inline float4 toFloat4() const
    {
        return Float3_int{f, i}.toFloat4();
    }
};

__HD__ void inline operator+=(Force& a, const Force& b)
{
    a.f.x += b.f.x;
    a.f.y += b.f.y;
    a.f.z += b.f.z;
}    

__HD__ Force inline operator+(Force a, const Force& b)
{
    a += b;
    return a;
}


struct Stress
{
    float xx, xy, xz, yy, yz, zz;
};

__HD__ void inline operator+=(Stress& a, const Stress& b)
{
    a.xx += b.xx; a.xy += b.xy; a.xz += b.xz;
    a.yy += b.yy; a.yz += b.yz; a.zz += b.zz;
}    

__HD__ Stress inline operator+(Stress a, const Stress& b)
{
    a += b;
    return a;
}


#define RIGID_MOTIONS_DOUBLE

#ifdef RIGID_MOTIONS_DOUBLE
using RigidReal  = double;
#else
using RigidReal  = float;
#endif

using RigidReal3 = VecTraits::Vec<RigidReal, 3>::Type;
using RigidReal4 = VecTraits::Vec<RigidReal, 4>::Type;

//=================================================================

template <class real>
struct __align__(16) TemplRigidMotion
{
    using R3 = typename VecTraits::Vec<real, 3>::Type;
    using R4 = typename VecTraits::Vec<real, 4>::Type;
    
    R3 r;
    R4 q;
    R3 vel, omega;
    R3 force, torque;
};

using DoubleRigidMotion = TemplRigidMotion<double>;
using SingleRigidMotion = TemplRigidMotion<float>;
using RigidMotion       = TemplRigidMotion<RigidReal>;

struct __align__(16) COMandExtent
{
    float3 com, low, high;
};
