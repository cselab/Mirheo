#pragma once

// http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
// https://arxiv.org/pdf/0811.2889.pdf

__device__ __host__ __forceinline__ float4 f3toQ(const float3 vec)
{
	return {0.0f, vec.x, vec.y, vec.z};
}

__device__ __host__ __forceinline__ double4 f3toQ(const double3 vec)
{
	return {0.0f, vec.x, vec.y, vec.z};
}

template<class R4>
__device__ __host__ __forceinline__ R4 invQ(const R4 q)
{
	return {q.x, -q.y, -q.z, -q.w};
}

template<class R4>
__device__ __host__ __forceinline__ R4 multiplyQ(const R4 q1, const R4 q2)
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
__device__ __host__ __forceinline__ R3 rotate(const R3 x, const R4 q)
{
	R4 qX = { (decltype(q.x))0.0f,
			  (decltype(q.x))x.x,
			  (decltype(q.x))x.y,
			  (decltype(q.x))x.z };

	qX = multiplyQ(multiplyQ(q, qX), invQ(q));

	return { (decltype(x.x))qX.y,
			 (decltype(x.x))qX.z,
			 (decltype(x.x))qX.w };
}

template<class R4, class R3>
__device__ __host__ __forceinline__ R4 compute_dq_dt(const R4 q, const R3 omega)
{
	return 0.5f*multiplyQ(f3toQ(omega), q);
}

