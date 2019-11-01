#pragma once

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/vec_traits.h>

namespace Quaternion
{
// http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
// https://arxiv.org/pdf/0811.2889.pdf

template <typename R3>
__HD__ inline auto f3toQ(const R3& vec)
{
    using RealType  = decltype(vec.x);
    using RealType4 = typename VecTraits::Vec<RealType, 4>::Type;
    
    return RealType4 {static_cast<RealType>(0.0), vec.x, vec.y, vec.z};
}

template<class R4>
__HD__ inline R4 conjugate(const R4 q)
{
    return {q.x, -q.y, -q.z, -q.w};
}

// https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
template <typename R3>
__HD__ inline auto getFromVectorPair(R3 u, R3 v)
{
    using RealType  = decltype(u.x);
    using RealType4 = typename VecTraits::Vec<RealType, 4>::Type;

    auto kCosTheta = dot(u, v);
    auto k = math::sqrt(dot(u,u) * dot(v,v));

    if (math::abs(kCosTheta + k) < 1e-6)
    {
        // 180 degree rotation around any orthogonal vector
        return f3toQ( normalize(anyOrthogonal(u)) );
    }
    auto uv = cross(u, v);
    RealType4 q {kCosTheta + k, uv.x, uv.y, uv.z};
    return normalize(q);
}


template<class R4>
__HD__ inline R4 multiply(const R4 q1, const R4 q2)
{
    R4 res;
    res.x =  q1.x * q2.x - q1.y * q2.y - q1.z * q2.z - q1.w * q2.w;
    res.y =  q1.x * q2.y + q1.y * q2.x + q1.z * q2.w - q1.w * q2.z;
    res.z =  q1.x * q2.z - q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
    res.w =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
    return res;
}

// rotate a point v in 3D space around the origin using this quaternion
template<class R4, class R3>
__HD__ inline R3 rotate(const R3 x, const R4 q)
{
    using QRealType = decltype(R4::x);
    using VRealType = decltype(R3::x);
    
    R4 qX = { static_cast<QRealType>(0.0),
              static_cast<QRealType>(x.x),
              static_cast<QRealType>(x.y),
              static_cast<QRealType>(x.z) };

    qX = multiply(multiply(q, qX), conjugate(q));

    return { static_cast<VRealType>(qX.y),
             static_cast<VRealType>(qX.z),
             static_cast<VRealType>(qX.w) };
}

template<class R4, class R3>
__HD__ inline R4 timeDerivative(const R4 q, const R3 omega)
{
    using RealType = decltype(R4::x);
    constexpr RealType half = static_cast<RealType>(0.5);
    return half * multiply(f3toQ(omega), q);
}

} // namespace Quaternion
