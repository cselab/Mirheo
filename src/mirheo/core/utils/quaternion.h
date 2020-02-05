#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/vec_traits.h>

namespace mirheo
{

// http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
// https://arxiv.org/pdf/0811.2889.pdf

template<class Real>
class __align__(16) Quaternion
{
 public:
    using Real3 = typename VecTraits::Vec<Real, 3>::Type;
    using Real4 = typename VecTraits::Vec<Real, 4>::Type;

    __HD__ static inline Quaternion createFromComponents(Real w, Real x, Real y, Real z)
    {
        return {w, x, y, z};
    }

    __HD__ static inline Quaternion createFromComponents(Real w, Real3 v)
    {
        return {w, v};
    }

    __HD__ static inline Quaternion createFromComponents(Real4 v)
    {
        return {v.x, v.y, v.z, v.w};
    }

    __HD__ static inline Quaternion pureVector(Real3 v)
    {
        return {static_cast<Real>(0), v.x, v.y, v.z};
    }

    __HD__ static inline Quaternion createFromRotation(Real angle, Real3 axis)
    {
        const Real alpha = static_cast<Real>(0.5) * angle;
        const Real3 u = math::rsqrt(dot(axis, axis)) * axis;
        return {std::cos(alpha), std::sin(alpha) * u};
    }

    __HD__ static inline Quaternion createFromVectors(Real3 from, Real3 to)
    {
        return {from, to};
    }

    Quaternion() = default;
    Quaternion(const Quaternion& q) = default;
    Quaternion& operator=(const Quaternion& q) = default;
    ~Quaternion() = default;


    template <class T>
    __HD__ explicit operator Quaternion<T>() const
    {
        return Quaternion<T>::createFromComponents(static_cast<T>(w),
                                                   static_cast<T>(x),
                                                   static_cast<T>(y),
                                                   static_cast<T>(z));
    }

    __HD__ explicit operator float4() const
    {
        return {static_cast<float>(w),
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(z)};
    }

    __HD__ explicit operator double4() const
    {
        return {static_cast<double>(w),
                static_cast<double>(x),
                static_cast<double>(y),
                static_cast<double>(z)};
    }
        
    __HD__ inline Real realPart() const {return w;}
    __HD__ inline Real3 vectorPart() const {return {x, y, z};}

    __HD__ inline Quaternion<Real> conjugate() const {return {w, -x, -y, -z};}

    __HD__ inline Real norm() const {return math::sqrt(w*w + x*x + y*y + z*z);}

    __HD__ inline Quaternion& normalize()
    {
        const Real factor = math::rsqrt(w*w + x*x + y*y + z*z);
        return *this *= factor;
    }
    
    __HD__ inline Quaternion normalized() const
    {
        Quaternion ret = *this;
        ret.normalize();
        return ret;
    }

    __HD__ inline Quaternion& operator+=(const Quaternion& q)
    {
        x += q.x;
        y += q.y;
        z += q.z;
        w += q.w;
        return *this;
    }

    __HD__ inline Quaternion& operator-=(const Quaternion& q)
    {
        x -= q.x;
        y -= q.y;
        z -= q.z;
        w -= q.w;
        return *this;
    }

    __HD__ inline Quaternion& operator*=(Real a)
    {
        x *= a;
        y *= a;
        z *= a;
        w *= a;
        return *this;
    }

    __HD__ friend inline Quaternion operator+(Quaternion q1, const Quaternion& q2) {q1 += q2; return q1;}
    __HD__ friend inline Quaternion operator-(Quaternion q1, const Quaternion& q2) {q1 -= q2; return q1;}

    __HD__ friend inline Quaternion operator*(Real a, Quaternion q) {q *= a; return q;}
    __HD__ friend inline Quaternion operator*(Quaternion q, Real a) {return a * q;}

    __HD__ friend inline Quaternion operator*(const Quaternion& q1, const Quaternion& q2)
    {
        return {q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
                q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
                q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
                q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w};
    }
    
    __HD__ inline Quaternion& operator*=(const Quaternion& q)
    {
        *this = (*this) * q;
        return *this;
    }

    __HD__ inline Real3 rotate(Real3 v) const
    {
        const auto qv = pureVector(v);
        const auto& q = *this;
        return (q * qv * q.conjugate()).vectorPart();
    }

    __HD__ inline Real3 inverseRotate(Real3 v) const
    {
        const auto qv = pureVector(v);
        const auto& q = *this;
        return (q.conjugate() * qv * q).vectorPart();
    }

    __HD__ inline Quaternion timeDerivative(Real3 omega) const
    {
        constexpr Real half = static_cast<Real>(0.5);
        return half * pureVector(omega) * *this;
    }

    
 public:
    
    Real w;       // real part
    Real x, y, z; // vector part
    
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
    
    // https://stackoverflow.com/a/11741520/11630848
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
