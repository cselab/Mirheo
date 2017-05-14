#pragma once

template<typename T>
__inline__ __device__ T sqr(T val)
{
	return val*val;
}

template<typename Operation>
__inline__ __device__ float3 warpReduce(float3 val, Operation op)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = op(val.x, __shfl_down(val.x, offset));
		val.y = op(val.y, __shfl_down(val.y, offset));
		val.z = op(val.z, __shfl_down(val.z, offset));
	}
	return val;
}

__device__ __forceinline__ float4 readNoCache(const float4* addr)
{
	float4 res;
	asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(res.x), "=f"(res.y), "=f"(res.z), "=f"(res.w) : "l"(addr));
	return res;
}

__device__ __forceinline__ void writeNoCache(float4* addr, const float4 val)
{
	asm("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
}
