#pragma once

// http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
// https://arxiv.org/pdf/0811.2889.pdf
__device__ __host__ __forceinline__ float4 f3toQ(const float3 vec)
{
	return make_float4(0.0f, vec.x, vec.y, vec.z);
}
__device__ __host__ __forceinline__ float4 invQ(const float4 q)
{
	return make_float4(q.x, -q.y, -q.z, -q.w);
}

__device__ __host__ __forceinline__ float4 multiplyQ(const float4 q1, const float4 q2)
{
	float4 res;
	res.x =  q1.x * q2.x - q1.y * q2.y - q1.z * q2.z - q1.w * q2.w;
	res.y =  q1.x * q2.y + q1.y * q2.x + q1.z * q2.w - q1.w * q2.z;
	res.z =  q1.x * q2.z - q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
	res.w =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
	return res;
}

// rotate a point v in 3D space around the origin using this quaternion
__device__ __host__ __forceinline__ float3 rotate(const float3 x, const float4 q)
{
	float4 qX = make_float4(0.0f, x);
	qX = multiplyQ(multiplyQ(q, qX), invQ(q));

	return make_float3(qX.y, qX.z, qX.w);
}

__device__ __host__ __forceinline__ float4 compute_dq_dt(const float4 q, const float3 omega)
{
	return 0.5f*multiplyQ(f3toQ(omega), q);
}
