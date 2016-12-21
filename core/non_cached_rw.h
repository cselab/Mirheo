#pragma once

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
