#pragma once

#include "helper_math.h"

#ifdef __CDT_PARSER__
using cudaStream_t = int;
const int warpSize = 32;
#endif

inline int getNblocks(const int n, const int nthreads)
{
	return (n+nthreads-1) / nthreads;
}

__host__ __device__ __forceinline__ float3 f4tof3(float4 x)
{
	return make_float3(x.x, x.y, x.z);
}

template<typename T>
__host__ __device__ __forceinline__  T sqr(T val)
{
	return val*val;
}

//=======================================================================================
// Per-warp reduction operations
//=======================================================================================

//****************************************************************************
// float
//****************************************************************************
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

template<typename Operation>
__device__ __forceinline__  float2 warpReduce(float2 val, Operation op)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = op(val.x, __shfl_down(val.x, offset));
		val.y = op(val.y, __shfl_down(val.y, offset));
	}
	return val;
}

template<typename Operation>
__device__ __forceinline__  float warpReduce(float val, Operation op)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val = op(val, __shfl_down(val, offset));
	}
	return val;
}

//****************************************************************************
// double
//****************************************************************************

template<typename Operation>
__device__ __forceinline__  double3 warpReduce(double3 val, Operation op)
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

template<typename Operation>
__device__ __forceinline__  double2 warpReduce(double2 val, Operation op)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = op(val.x, __shfl_down(val.x, offset));
		val.y = op(val.y, __shfl_down(val.y, offset));
	}
	return val;
}

template<typename Operation>
__device__ __forceinline__  double warpReduce(double val, Operation op)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val = op(val, __shfl_down(val, offset));
	}
	return val;
}

//****************************************************************************
// int
//****************************************************************************

template<typename Operation>
__device__ __forceinline__  int3 warpReduce(int3 val, Operation op)
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

template<typename Operation>
__device__ __forceinline__  int2 warpReduce(int2 val, Operation op)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x = op(val.x, __shfl_down(val.x, offset));
		val.y = op(val.y, __shfl_down(val.y, offset));
	}
	return val;
}

template<typename Operation>
__device__ __forceinline__  int warpReduce(int val, Operation op)
{
#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val = op(val, __shfl_down(val, offset));
	}
	return val;
}

//=======================================================================================
// Atomics for vector types
//=======================================================================================


__device__ __forceinline__ float2 atomicAdd(float2* addr, float2 v)
{
	float2 res;
	res.x = atomicAdd((float*)addr,   v.x);
	res.y = atomicAdd((float*)addr+1, v.y);
	return res;
}

__device__ __forceinline__ float3 atomicAdd(float3* addr, float3 v)
{
	float3 res;
	res.x = atomicAdd((float*)addr,   v.x);
	res.y = atomicAdd((float*)addr+1, v.y);
	res.z = atomicAdd((float*)addr+2, v.z);
	return res;
}

__device__ __forceinline__ float3 atomicAdd(float4* addr, float3 v)
{
	float3 res;
	res.x = atomicAdd((float*)addr,   v.x);
	res.y = atomicAdd((float*)addr+1, v.y);
	res.z = atomicAdd((float*)addr+2, v.z);
	return res;
}

__device__ __forceinline__ double3 atomicAdd(double3* addr, double3 v)
{
	double3 res;
	res.x = atomicAdd((double*)addr,   v.x);
	res.y = atomicAdd((double*)addr+1, v.y);
	res.z = atomicAdd((double*)addr+2, v.z);
	return res;
}

//=======================================================================================
// Read/write through cache
//=======================================================================================

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

//=======================================================================================
// Lane and warp id
// https://stackoverflow.com/questions/28881491/how-can-i-find-out-which-thread-is-getting-executed-on-which-core-of-the-gpu
//=======================================================================================

__device__ __forceinline__ uint32_t __warpid()
{
	uint32_t warpid;
	asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
	return warpid;
}

__device__ __forceinline__ uint32_t __laneid()
{
	uint32_t laneid;
	asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
	return laneid;
}

//=======================================================================================
// Warp-aggregated atomic increment
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
//=======================================================================================

template<int DIMS>
__device__ __forceinline__ uint getLaneId();

template<>
__device__ __forceinline__ uint getLaneId<1>()
{
	return threadIdx.x & 31;
}

template<>
__device__ __forceinline__ uint getLaneId<2>()
{
	return ((threadIdx.y * blockDim.x) + threadIdx.x) & 31;
}

template<>
__device__ __forceinline__ uint getLaneId<3>()
{
	return (threadIdx.z * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) & 31;
}

template<int DIMS=1>
__device__ __forceinline__ int atomicAggInc(int *ctr)
{
	int lane_id = getLaneId<DIMS>();

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


