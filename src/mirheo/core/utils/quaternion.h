#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/vec_traits.h>

namespace mirheo
{


/** \brief Quaternion representation with basic operations.
    \tparam Real The precision to be used. Must be a scalar real number type (e.g. float, double).

    See also:
    - http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    - https://arxiv.org/pdf/0811.2889.pdf
*/
template<class Real>
class __align__(16) Quaternion
{
 public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // breathe warnings
    using Real3 = typename vec_traits::Vec<Real, 3>::Type; ///< real3
    using Real4 = typename vec_traits::Vec<Real, 4>::Type; ///< real4
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// \brief Create a \c Quaternion from components
    __HD__ static inline Quaternion createFromComponents(Real w, Real x, Real y, Real z)
    {
        return {w, x, y, z};
    }

    /// \brief Create a \c Quaternion from real part and vector part
    __HD__ static inline Quaternion createFromComponents(Real w, Real3 v)
    {
        return {w, v};
    }

    /// \brief Create a \c Quaternion from components
    __HD__ static inline Quaternion createFromComponents(Real4 v)
    {
        return {v.x, v.y, v.z, v.w};
    }

    /// \brief Create a pure vector \c Quaternion
    __HD__ static inline Quaternion pureVector(Real3 v)
    {
        return {static_cast<Real>(0), v.x, v.y, v.z};
    }

    /** \brief Create a \c Quaternion that represents the rotation around an axis with a given angle
        \param angle The angle (in radians) of the rotation
        \param axis The axis of rotation, must be non zero (or nan will be returned)
     */
    __HD__ static inline Quaternion createFromRotation(Real angle, Real3 axis)
    {
        const Real alpha = static_cast<Real>(0.5) * angle;
        const Real3 u = math::rsqrt(dot(axis, axis)) * axis;
        return {std::cos(alpha), std::sin(alpha) * u};
    }

    /** \brief Create a \c Quaternion that represents the "shortest" rotation between two vectors
        \param [in] from The origin vector
        \param [in] to The vector obtained by applying the rotation to \p from

        The vectors must be non zero.
     */
    __HD__ static inline Quaternion createFromVectors(Real3 from, Real3 to)
    {
        return {from, to};
    }

    Quaternion() = default;
    Quaternion(const Quaternion& q) = default; ///< copy constructor
    Quaternion& operator=(const Quaternion& q) = default; ///< assignment operator
    ~Quaternion() = default;


    /// conversion to different precision
    template <class T>
    __HD__ explicit operator Quaternion<T>() const
    {
        return Quaternion<T>::createFromComponents(static_cast<T>(w),
                                                   static_cast<T>(x),
                                                   static_cast<T>(y),
                                                   static_cast<T>(z));
    }

    /// conversion to float4 (real part will be stored first, followed by the vector part)
    __HD__ explicit operator float4() const
    {
        return {static_cast<float>(w),
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(z)};
    }

    /// conversion to double4 (real part will be stored first, followed by the vector part)
    __HD__ explicit operator double4() const
    {
        return {static_cast<double>(w),
                static_cast<double>(x),
                static_cast<double>(y),
                static_cast<double>(z)};
    }

    __HD__ inline Real realPart() const {return w;} ///< \return the real part of the quaternion
    __HD__ inline Real3 vectorPart() const {return {x, y, z};} ///< \return the vector part of the quaternion

    __HD__ inline Quaternion<Real> conjugate() const {return {w, -x, -y, -z};} ///< \return the conjugate of the quaternion

    __HD__ inline Real norm() const {return math::sqrt(w*w + x*x + y*y + z*z);}  ///< \return the norm of the quaternion

    /// Normalize the current quaternion. Must be non zero
    __HD__ inline Quaternion& normalize()
    {
        const Real factor = math::rsqrt(w*w + x*x + y*y + z*z);
        return *this *= factor;
    }

    /// \return A normalized copy of this Quaternion
    __HD__ inline Quaternion normalized() const
    {
        Quaternion ret = *this;
        ret.normalize();
        return ret;
    }

    /// Add a quaternion to the current one
    __HD__ inline Quaternion& operator+=(const Quaternion& q)
    {
        x += q.x;
        y += q.y;
        z += q.z;
        w += q.w;
        return *this;
    }

    /// Subtract a quaternion to the current one
    __HD__ inline Quaternion& operator-=(const Quaternion& q)
    {
        x -= q.x;
        y -= q.y;
        z -= q.z;
        w -= q.w;
        return *this;
    }

    /// Scale the current quaternion
    __HD__ inline Quaternion& operator*=(Real a)
    {
        x *= a;
        y *= a;
        z *= a;
        w *= a;
        return *this;
    }

    /// \return The sum of 2 quaternions
    __HD__ friend inline Quaternion operator+(Quaternion q1, const Quaternion& q2) {q1 += q2; return q1;}
    /// \return The difference of 2 quaternions
    __HD__ friend inline Quaternion operator-(Quaternion q1, const Quaternion& q2) {q1 -= q2; return q1;}

    /// \return The scalar multiplication of a quaternion
    __HD__ friend inline Quaternion operator*(Real a, Quaternion q) {q *= a; return q;}

    /// \return The scalar multiplication of a quaternion
    __HD__ friend inline Quaternion operator*(Quaternion q, Real a) {return a * q;}

    /// \return The quaternion product of 2 quaternions
    __HD__ friend inline Quaternion operator*(const Quaternion& q1, const Quaternion& q2)
    {
        return {q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
                q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
                q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
                q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w};
    }

    /// Multiply the current quaternion with another with Quaternion multiplication and store the result in this object
    __HD__ inline Quaternion& operator*=(const Quaternion& q)
    {
        *this = (*this) * q;
        return *this;
    }

    /// \return The input vector rotated by the current quaternion
    __HD__ inline Real3 rotate(Real3 v) const
    {
        const auto qv = pureVector(v);
        const auto& q = *this;
        return (q * qv * q.conjugate()).vectorPart();
    }

    /// \return The input vector rotated by the current quaternion inverse
    __HD__ inline Real3 inverseRotate(Real3 v) const
    {
        const auto qv = pureVector(v);
        const auto& q = *this;
        return (q.conjugate() * qv * q).vectorPart();
    }

    /// \return The time derivative of the given angular velocity, useful for time integration of rigid objects
    __HD__ inline Quaternion timeDerivative(Real3 omega) const
    {
        constexpr Real half = static_cast<Real>(0.5);
        return half * pureVector(omega) * *this;
    }


 public:
    Real w; ///< real part
    Real x; ///< vector part, x
    Real y; ///< vector part, y
    Real z; ///< vector part, z

 private:
    __HD__ Quaternion(Real rw, Real vx, Real vy, Real vz) :
        w(rw),
        x(vx),
        y(vy),
        z(vz)
    {}

    __HD__ Quaternion(Real rw, Real3 u) :
        w(rw),
        x(u.x),
        y(u.y),
        z(u.z)
    {}

    /** Create a Quaternion from two vectors.
        The quaternion representa the direct rotation that transform u into v.
        See also  https://stackoverflow.com/a/11741520/11630848
    */
    __HD__ Quaternion(Real3 u, Real3 v)
    {
        const Real k_cos_theta = dot(u, v);
        const Real k = math::sqrt(dot(u, u) * dot(v, v));

        if (math::abs(k_cos_theta + k) < 1e-6) // opposite directions
        {
            w = static_cast<Real>(0);
            const Real3 n = anyOrthogonal(u);
            x = n.x;
            y = n.y;
            z = n.z;
        }
        else
        {
            w = k_cos_theta + k;
            const Real3 n = cross(u, v);
            x = n.x;
            y = n.y;
            z = n.z;
        }
        this->normalize();
    }
};

} // namespace mirheo
