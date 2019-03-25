#pragma once

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>

// http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
// https://arxiv.org/pdf/0811.2889.pdf

__HD__ inline float4 f3toQ(const float3 vec)
{
    return {0.0f, vec.x, vec.y, vec.z};
}

__HD__ inline double4 f3toQ(const double3 vec)
{
    return {0.0f, vec.x, vec.y, vec.z};
}

template<class R4>
__HD__ inline R4 invQ(const R4 q)
{
    return {q.x, -q.y, -q.z, -q.w};
}

// https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
__HD__ inline float4 getQfrom(float3 u, float3 v)
{
    auto kCosTheta = dot(u, v);
    auto k = sqrtf(dot(u,u) * dot(v,v));

    if (fabs(kCosTheta + k) < 1e-6)
    {
        // 180 degree rotation around any orthogonal vector
        return f3toQ( normalize(anyOrthogonal(u)) );
    }
    auto uv = cross(u, v);
    float4 q {kCosTheta + k, uv.x, uv.y, uv.z};
    return normalize(q);
}

template<class R4>
__HD__ inline R4 multiplyQ(const R4 q1, const R4 q2)
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
    using Qreal = decltype(R4::x);
    using Vreal = decltype(R3::x);
    
    R4 qX = { (Qreal)0.0,
              (Qreal)x.x,
              (Qreal)x.y,
              (Qreal)x.z };

    qX = multiplyQ(multiplyQ(q, qX), invQ(q));

    return { (Vreal)qX.y,
             (Vreal)qX.z,
             (Vreal)qX.w };
}

template<class R4, class R3>
__HD__ inline R4 compute_dq_dt(const R4 q, const R3 omega)
{
    return 0.5f*multiplyQ(f3toQ(omega), q);
}

