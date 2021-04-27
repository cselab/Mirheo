// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/vec_traits.h>

#include <cstdint>
#include <type_traits>

namespace mirheo
{

#ifdef MIRHEO_DOUBLE_PRECISION
using real    = double;  ///< represents a scalar real number
using integer = int64_t; ///< represents an integer number
#define MIRHEO_SCNgREAL "lg"  ///< scanf format for real, see `<cinttypes>`
#define MIRHEO_PRIgREAL "g"   ///< printf format for real
#else
using real    = float;   ///< represents a scalar real number
using integer = int32_t; ///< represents an integer number
#define MIRHEO_SCNgREAL "g"   ///< scanf format for real, see `<cinttypes>`
#define MIRHEO_PRIgREAL "g"   ///< printf format for real
#endif

using real2 = vec_traits::Vec<real, 2>::Type; ///< a pair of real numbers
using real3 = vec_traits::Vec<real, 3>::Type; ///< three real numbers
using real4 = vec_traits::Vec<real, 4>::Type; ///< faour real numbers

inline namespace unit_literals
{
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

/** \brief Helper class for packing/unpacking \e real3 + \e integer into \e real4
 */
struct __align__(16) Real3_int
{
    real3 v;   ///< vector part
    integer i; ///< integer part

    /** \brief A special value used to mark particles.

        Marked particles will be deleted during cell list rebuild. For objects,
        objects with all particles marked will be removed during object
        redistribution.
    */
    static constexpr real mark_val = -900._r;

    /// copy constructor
    __HD__ inline Real3_int(const Real3_int& x)
    {
        *reinterpret_cast<real4*>(this) = *reinterpret_cast<const real4*>(&x);
    }

    /// assignment operator
    __HD__ inline Real3_int& operator=(const Real3_int& x)
    {
        *reinterpret_cast<real4*>(this) = *reinterpret_cast<const real4*>(&x);
        return *this;
    }

    /// defult constructor; NO default values!
    __HD__ inline Real3_int() {}

    /// Constructor from vector and integer
    __HD__ inline Real3_int(real3 vecPart, integer intPart) :
        v(vecPart),
        i(intPart)
    {}

    /// Constructor from 4 components vector; the last one will be reinterpreted to integer (not converted)
    __HD__ inline Real3_int(const real4 r4)
    {
        *reinterpret_cast<real4*>(this) = r4;
    }

    /// \return reinterpreted values packed in a real4 (no conversion)
    __HD__ inline real4 toReal4() const
    {
        return *reinterpret_cast<const real4*>(this);
    }

    /// Mark this object; see isMarked().
    /// Does not modify the integer part
    __HD__ inline void mark()
    {
        v.x = v.y = v.z = mark_val;
    }

    /// \return \c true if the object has been marked via mark()
    __HD__ inline bool isMarked() const
    {
        return
            v.x == mark_val &&
            v.y == mark_val &&
            v.z == mark_val;
    }
};

/** \brief Structure that holds position, velocity and global index of one particle.

    Due to performance reasons it should be aligned to 16 bytes boundary,
    therefore 8 bytes = 2 integer numbers are extra.
    The integer fields are used to store the global index
 */
struct __align__(16) Particle
{
    real3 r;     ///< position
    integer i1;  ///< lower part of particle id

    real3 u;         ///< velocity
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

    /** \brief Default constructor

        \rst
        .. attention::
            The default constructor DOES NOT initialize any members!
        \endrst
     */
    __HD__ inline Particle() {};

    /// Set the global index of the particle
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

    /// \return the global index of the particle
    __HD__ inline int64_t getId() const
    {
#ifdef MIRHEO_DOUBLE_PRECISION
        return i1;
#else
        return int64_t(i1) + (int64_t (i2) << 32);
#endif
    }

    /** \brief Construct a Particle from two real4 entries
        \param r4 first three reals will be position (#r), last one \e \.w - #i1 (reinterpreted, not converted)
        \param u4 first three reals will be velocity (#u), last one \e \.w - #i2 (reinterpreted, not converted)
     */
    __HD__ inline Particle(const real4 r4, const real4 u4)
    {
        Real3_int rtmp(r4), utmp(u4);
        r  = rtmp.v;
        i1 = rtmp.i;
        u  = utmp.v;
        i2 = utmp.i;
    }

    /** \brief read position from array and stores it internally
        \param addr start of the array with size > \p pid
        \param pid  particle index
     */
    __HD__ inline void readCoordinate(const real4 *addr, const int pid)
    {
        const Real3_int tmp = addr[pid];
        r  = tmp.v;
        i1 = tmp.i;
    }

    /** \brief read velocity from array and stores it internally
        \param addr pointer to the start of the array. Must be larger than \p pid
        \param pid  particle index
     */
    __HD__ inline void readVelocity(const real4 *addr, const int pid)
    {
        const Real3_int tmp = addr[pid];
        u  = tmp.v;
        i2 = tmp.i;
    }

    /// \return packed #r and #i1 as \e Real3_int
    __HD__ inline Real3_int r2Real3_int() const
    {
        return Real3_int{r, i1};
    }

    /** \brief Helps writing particles back to \e real4 array
        \return packed #r and #i1 as \e real4
     */
    __HD__ inline real4 r2Real4() const
    {
        return r2Real3_int().toReal4();
    }

    /// \return packed #u and #i2 as \e Real3_int
    __HD__ inline Real3_int u2Real3_int() const
    {
        return Real3_int{u, i2};
    }

    /** \brief Helps writing particles back to \e real4 array
        \return packed #u and #i2 as \e real4
     */
    __HD__ inline real4 u2Real4() const
    {
        return u2Real3_int().toReal4();
    }

    /** \brief Helps writing particles back to \e real4 arrays
        \param pos destination array that contains positions
        \param vel destination array that contains velocities
        \param pid particle index
     */
    __HD__ inline void write2Real4(real4 *pos, real4 *vel, int pid) const
    {
        pos[pid] = r2Real4();
        vel[pid] = u2Real4();
    }

    /// mark the particle; this will erase its position information
    __HD__ inline void mark()
    {
        Real3_int f3i = r2Real3_int();
        f3i.mark();
        r = f3i.v;
    }

    /// \return \c true if the particle has been marked
    __HD__ inline bool isMarked() const
    {
        return r2Real3_int().isMarked();
    }
};

/** \brief Structure that holds force as \e real4 (to reduce number of load/store instructions)

    Due to performance reasons it should be aligned to 16 bytes boundary.
    The integer field is not reserved for anything at the moment
 */
struct __align__(16) Force
{
    real3 f;   ///< Force value
    integer i; ///< extra integer variable (unused)

    /// default constructor, does NOT initialize anything
    __HD__ inline Force() {};

    /// Construct a \c Force from a vector part and an integer part
    __HD__ inline Force(const real3 vecPart, int intPart) :
        f(vecPart),
        i(intPart)
    {};

    /// Construct a force from \c real4.
    /// The 4th component will be reinterpreted as an integer (not converted)
    __HD__ inline Force(const real4 f4)
    {
        Real3_int tmp(f4);
        f = tmp.v;
        i = tmp.i;
    }

    /// \return packed real part + integer part as \e real4
    __HD__ inline real4 toReal4() const
    {
        return Real3_int{f, i}.toReal4();
    }
};

/// add b to a (the integer part is ignored)
__HD__ static inline void operator+=(Force& a, const Force& b)
{
    a.f.x += b.f.x;
    a.f.y += b.f.y;
    a.f.z += b.f.z;
}

/// \return a + b (the integer part is ignored)
__HD__ static inline Force operator+(Force a, const Force& b)
{
    a += b;
    return a;
}

/** \brief Store a symmetric stess tensor in 3 dimensions

    Since it is symmetric, only 6 components are needed (diagonal and upper part
 */
struct Stress
{
    real xx; ///< x diagonal term
    real xy; ///< xy upper term
    real xz; ///< xz upper term
    real yy; ///< y diagonal term
    real yz; ///< yz upper term
    real zz; ///< z diagonal term
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

/// Contains the rigid object center of mass and bounding box
/// Used e.g. to decide which domain the objects belong to and
/// what particles / cells are close to it
struct __align__(16) COMandExtent
{
    real3 com;  ///< center of mass
    real3 low;  ///< lower corner of the bounding box
    real3 high; ///< upper corner of the bounding box
};

/// Contains coordinates of the center of mass and orientation of an object
/// Used to initialize object positions
struct ComQ
{
    real3 r; ///< object position
    real4 q; ///< quaternion that represents the orientation
};

} // namespace mirheo
