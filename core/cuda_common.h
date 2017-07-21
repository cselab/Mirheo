#pragma once

#include <core/helper_math.h>

// Workaround for nsight
#ifndef __CUDACC_EXTENDED_LAMBDA__
#define __device__
#endif

inline int getNblocks(const int n, const int nthreads)
{
	return (n+nthreads-1) / nthreads;
}

template<typename T>
__host__ __device__ __forceinline__  T sqr(T val)
{
	return val*val;
}

template<typename Operation>
__device__ __forceinline__  float3 warpReduce(float3 val, Operation op)
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

__device__ __forceinline__ float3 atomicAdd(float3* addr, float3 v)
{
	float3 res;
	res.x = atomicAdd((float*)addr,   v.x);
	res.y = atomicAdd((float*)addr+1, v.y);
	res.z = atomicAdd((float*)addr+2, v.z);
	return res;
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

// warp-aggregated atomic increment
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ __forceinline__ int atomicAggInc(int *ctr)
{
	int lane_id = (threadIdx.x % 32);

	int mask = __ballot(1);
	// select the leader
	int leader = __ffs(mask) - 1;
	// leader does the update
	int res;
	if(lane_id == leader)
	res = atomicAdd(ctr, __popc(mask));
	// broadcast result
	res = __shfl(res, leader);
	// each thread computes its own value
	return res + __popc(mask & ((1 << lane_id) - 1));
}
