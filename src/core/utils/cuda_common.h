#pragma once

#include <tuple>
#include "helper_math.h"

#ifdef __CDT_PARSER__
using cudaStream_t = int;
const int warpSize = 32;

int3 blockIdx, blockDim, threadIdx;
#endif

inline int getNblocks(const int n, const int nthreads)
{
	return (n+nthreads-1) / nthreads;
}

__host__ __device__ inline float3 f4tof3(float4 x)
{
	return make_float3(x.x, x.y, x.z);
}

template<typename T>
__host__ __device__ inline  T sqr(T val)
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
__device__ inline  float3 warpReduce(float3 val, Operation op)
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
__device__ inline  float2 warpReduce(float2 val, Operation op)
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
__device__ inline  float warpReduce(float val, Operation op)
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
__device__ inline  double3 warpReduce(double3 val, Operation op)
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
__device__ inline  double2 warpReduce(double2 val, Operation op)
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
__device__ inline  double warpReduce(double val, Operation op)
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
__device__ inline  int3 warpReduce(int3 val, Operation op)
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
__device__ inline  int2 warpReduce(int2 val, Operation op)
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
__device__ inline  int warpReduce(int val, Operation op)
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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
         old = atomicCAS(address_as_ull, assumed,
                          __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ inline float2 atomicAdd(float2* addr, float2 v)
{
	float2 res;
	res.x = atomicAdd((float*)addr,   v.x);
	res.y = atomicAdd((float*)addr+1, v.y);
	return res;
}

__device__ inline float3 atomicAdd(float3* addr, float3 v)
{
	float3 res;
	res.x = atomicAdd((float*)addr,   v.x);
	res.y = atomicAdd((float*)addr+1, v.y);
	res.z = atomicAdd((float*)addr+2, v.z);
	return res;
}

__device__ inline float3 atomicAdd(float4* addr, float3 v)
{
	float3 res;
	res.x = atomicAdd((float*)addr,   v.x);
	res.y = atomicAdd((float*)addr+1, v.y);
	res.z = atomicAdd((float*)addr+2, v.z);
	return res;
}

__device__ inline double3 atomicAdd(double3* addr, double3 v)
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

__device__ inline float4 readNoCache(const float4* addr)
{
	float4 res;
	asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(res.x), "=f"(res.y), "=f"(res.z), "=f"(res.w) : "l"(addr));
	return res;
}

__device__ inline void writeNoCache(float4* addr, const float4 val)
{
	asm("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
}

//=======================================================================================
// Lane and warp id
// https://stackoverflow.com/questions/28881491/how-can-i-find-out-which-thread-is-getting-executed-on-which-core-of-the-gpu
//=======================================================================================

__device__ inline uint32_t __warpid()
{
	uint32_t warpid;
	asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
	return warpid;
}

__device__ inline uint32_t __laneid()
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
__device__ inline uint getLaneId();

template<>
__device__ inline uint getLaneId<1>()
{
	return threadIdx.x & 31;
}

template<>
__device__ inline uint getLaneId<2>()
{
	return ((threadIdx.y * blockDim.x) + threadIdx.x) & 31;
}

template<>
__device__ inline uint getLaneId<3>()
{
	return (threadIdx.z * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) & 31;
}

template<int DIMS=1>
__device__ inline int atomicAggInc(int *ctr)
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



__device__ inline float fastPower(const float x, const float k)
{
	if (fabsf(k - 1.0f)   < 1e-6f) return x;
	if (fabsf(k - 0.5f)   < 1e-6f) return sqrtf(fabsf(x));
	if (fabsf(k - 0.25f)  < 1e-6f) return sqrtf(sqrtf(fabsf(x)));
	//if (fabsf(k - 0.125f) < 1e-6f) return sqrtf(sqrtf(sqrtf(fabsf(x))));

    return powf(fabsf(x), k);
}

// Tuple initializations

inline float3 make_float3(std::tuple<float, float, float> t3)
{
    return make_float3(std::get<0>(t3), std::get<1>(t3), std::get<2>(t3));
}


inline int3 make_int3(std::tuple<int, int, int> t3)
{
    return make_int3(std::get<0>(t3), std::get<1>(t3), std::get<2>(t3));
}



