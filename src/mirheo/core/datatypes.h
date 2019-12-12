#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/vec_traits.h>

#include <cstdint>
#include <type_traits>

namespace mirheo
{

#ifdef MIRHEO_DOUBLE_PRECISION
using real    = double;
using integer = int64_t;
#else
using real    = float;
using integer = int32_t;
#endif

using real2 = VecTraits::Vec<real, 2>::Type;
using real3 = VecTraits::Vec<real, 3>::Type;
using real4 = VecTraits::Vec<real, 4>::Type;

inline namespace unit_literals {
    __HD__ constexpr inline real operator "" _r (const long double a)
    {
        return static_cast<real>(a);
    }
} // namespace unit_literals

static inline __HD__ real2 make_real2(real x, real y)
{
    return {x, y};
}
static inline __HD__ real3 make_real3(real x, real y, real z)
{
    return {x, y, z};
}
static inline __HD__ real4 make_real4(real x, real y, real z, real w)
{
    return {x, y, z, w};
}


//==================================================================================================================
// Basic types
//==================================================================================================================

/**
 * Helper class for packing/unpacking \e float3 + \e int into \e float4
 */
struct __align__(16) Real3_int
{
    real3 v;
    integer i;

    static constexpr real mark_val = -900._r;

    __HD__ inline Real3_int(const Real3_int& x)
    {
        *reinterpret_cast<real4*>(this) = *reinterpret_cast<const real4*>(&x);
    }

    __HD__ inline Real3_int& operator=(const Real3_int& x)
    {
        *reinterpret_cast<real4*>(this) = *reinterpret_cast<const real4*>(&x);
        return *this;
    }

    __HD__ inline Real3_int() {}
    __HD__ inline Real3_int(real3 v, integer i) :
        v(v),
        i(i)
    {}

    __HD__ inline Real3_int(const real4 r4)
    {
        *reinterpret_cast<real4*>(this) = r4;
    }

    __HD__ inline real4 toReal4() const
    {
        return *reinterpret_cast<const real4*>(this);
    }

    __HD__ inline void mark()
    {
        v.x = v.y = v.z = mark_val;
    }

    __HD__ inline bool isMarked() const
    {
        return
            v.x == mark_val &&
            v.y == mark_val &&
            v.z == mark_val;
    }
};

/**
 * Structure to hold coordinates and velocities of particles.
 * Due to performance reasons it should be aligned to 16 bytes boundary,
 * therefore 8 bytes = 2 integer numbers are extra.
 * The integer fields are used for particle ids
 *
 * For performance reasons instead of an N-element array of Particle
 * an array of 2*N \e real4 elements is used.
 */
struct __align__(16) Particle
{
    real3 r;     ///< coordinate
    integer i1;  ///< lower part of particle id

    real3 u;     ///< velocity
    integer i2 {0};  ///< higher part of particle id

    /// Copy constructor uses efficient 16-bytes wide copies
    __HD__ inline Particle(const Particle& x)
    {
        auto r4this = reinterpret_cast<real4*>(this);
        auto r4x    = reinterpret_cast<const real4*>(&x);

        r4this[0] = r4x[0];
        r4this[1] = r4x[1];
    }

    /// Assignment operator uses efficient 16-bytes wide copies
    __HD__ inline Particle& operator=(Particle x)
    {
        auto r4this = reinterpret_cast<real4*>(this);
        auto r4x    = reinterpret_cast<const real4*>(&x);

        r4this[0] = r4x[0];
        r4this[1] = r4x[1];

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
#ifdef MIRHEO_DOUBLE_PRECISION
        i1 = id;
#else
        const int64_t highHalf = (id >> 32) << 32;
        i1 = (int32_t) (id - highHalf);
        i2 = (int32_t) (id >> 32);
#endif
    }

    __HD__ inline int64_t getId() const
    {
#ifdef MIRHEO_DOUBLE_PRECISION
        return i1;
#else
        return int64_t(i1) + (int64_t (i2) << 32);
#endif
    }
    
    /**
     * Construct a Particle from two real4 entries
     *
     * @param r4 first three reals will be coordinates (#r), last one, \e \.w - #i1
     * @param u4 first three reals will be velocities (#u), last one \e \.w - #i1
     */
    __HD__ inline Particle(const real4 r4, const real4 u4)
    {
        Real3_int rtmp(r4), utmp(u4);
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
    __HD__ inline void readCoordinate(const real4 *addr, const int pid)
    {
        const Real3_int tmp = addr[pid];
        r  = tmp.v;
        i1 = tmp.i;
    }

    /**
     * Only read velocities from the given \e addr
     *
     * @param addr must have at least \e pid entries
     * @param pid  particle id
     */
    __HD__ inline void readVelocity(const real4 *addr, const int pid)
    {
        const Real3_int tmp = addr[pid];
        u  = tmp.v;
        i2 = tmp.i;
    }

    __HD__ inline Real3_int r2Real3_int() const
    {
        return Real3_int{r, i1};
    }
    
    /**
     * Helps writing particles back to \e real4 array
     *
     * @return packed #r and #i1 as \e real4
     */
    __HD__ inline real4 r2Real4() const
    {
        return r2Real3_int().toReal4();
    }

    __HD__ inline Real3_int u2Real3_int() const
    {
        return Real3_int{u, i2};
    }

    /**
     * Helps writing particles back to \e real4 array
     *
     * @return packed #u and #i2 as \e real4
     */
    __HD__ inline real4 u2Real4() const
    {
        return u2Real3_int().toReal4();
    }

    /**
     * Helps writing particles back to \e real4 array
     *
     * @param dst must have at least \e 2*pid entries
     * @param pid particle id
     */
    __HD__ inline void write2Real4(real4 *pos, real4 *vel, int pid) const
    {
        pos[pid] = r2Real4();
        vel[pid] = u2Real4();
    }

    __HD__ inline void mark()
    {
        Real3_int f3i = r2Real3_int();
        f3i.mark();
        r = f3i.v;
    }

    __HD__ inline bool isMarked() const
    {
        return r2Real3_int().isMarked();
    }
};

struct __align__(16) Force
{
    real3 f;
    integer i;

    __HD__ inline Force() {};
    __HD__ inline Force(const real3 f, int i) :
        f(f),
        i(i)
    {};

    __HD__ inline Force(const real4 f4)
    {
        Real3_int tmp(f4);
        f = tmp.v;
        i = tmp.i;
    }

    __HD__ inline real4 toReal4() const
    {
        return Real3_int{f, i}.toReal4();
    }
};

__HD__ static inline void operator+=(Force& a, const Force& b)
{
    a.f.x += b.f.x;
    a.f.y += b.f.y;
    a.f.z += b.f.z;
}    

__HD__ static inline Force operator+(Force a, const Force& b)
{
    a += b;
    return a;
}


struct Stress
{
    real xx, xy, xz, yy, yz, zz;
};

__HD__ static inline void operator+=(Stress& a, const Stress& b)
{
    a.xx += b.xx; a.xy += b.xy; a.xz += b.xz;
    a.yy += b.yy; a.yz += b.yz; a.zz += b.zz;
}    

__HD__ static inline Stress operator+(Stress a, const Stress& b)
{
    a += b;
    return a;
}

struct __align__(16) COMandExtent
{
    real3 com, low, high;
};

struct ComQ
{
    real3 r;
    real4 q;
};

} // namespace mirheo
