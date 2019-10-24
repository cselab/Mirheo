#pragma once

#include "helper_math.h"
#include "cpu_gpu_defines.h"

static const cudaStream_t defaultStream = 0;

// shuffle instructions wrappers
#if __CUDACC_VER_MAJOR__ >= 9

#define MASK_ALL_WARP 0xFFFFFFFF

#define warpShfl(var, srcLane)     __shfl_sync      (MASK_ALL_WARP, var, srcLane)
#define warpShflDown(var, delta)   __shfl_down_sync (MASK_ALL_WARP, var, delta)
#define warpShflUp(var, delta)     __shfl_up_sync   (MASK_ALL_WARP, var, delta)
#define warpShflXor(var, laneMask) __shfl_xor_sync  (MASK_ALL_WARP, var, laneMask)
#define warpAll(predicate)         __all_sync       (MASK_ALL_WARP, predicate)
#define warpBallot(predicate)      __ballot_sync    (MASK_ALL_WARP, predicate)

#else

#define warpShfl(var, srcLane)     __shfl      (var, srcLane)
#define warpShflDown(var, delta)   __shfl_down (var, delta)
#define warpShflUp(var, delta)     __shfl_up   (var, delta)
#define warpShflXor(var, laneMask) __shfl_xor  (var, laneMask)
#define warpAll(predicate)         __all       (predicate)
#define warpBallot(predicate)      __ballot    (predicate)

#endif

inline int getNblocks(const int n, const int nthreads)
{
    return (n+nthreads-1) / nthreads;
}

template<typename T>
__HD__ inline  T sqr(T val)
{
    return val*val;
}

#ifdef __CUDACC__

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
        val.x = op(val.x, warpShflDown(val.x, offset));
        val.y = op(val.y, warpShflDown(val.y, offset));
        val.z = op(val.z, warpShflDown(val.z, offset));
    }
    return val;
}

template<typename Operation>
__device__ inline  float2 warpReduce(float2 val, Operation op)
{
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val.x = op(val.x, warpShflDown(val.x, offset));
        val.y = op(val.y, warpShflDown(val.y, offset));
    }
    return val;
}

template<typename Operation>
__device__ inline  float warpReduce(float val, Operation op)
{
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val = op(val, warpShflDown(val, offset));
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
        val.x = op(val.x, warpShflDown(val.x, offset));
        val.y = op(val.y, warpShflDown(val.y, offset));
        val.z = op(val.z, warpShflDown(val.z, offset));
    }
    return val;
}

template<typename Operation>
__device__ inline  double2 warpReduce(double2 val, Operation op)
{
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val.x = op(val.x, warpShflDown(val.x, offset));
        val.y = op(val.y, warpShflDown(val.y, offset));
    }
    return val;
}

template<typename Operation>
__device__ inline  double warpReduce(double val, Operation op)
{
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val = op(val, warpShflDown(val, offset));
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
        val.x = op(val.x, warpShflDown(val.x, offset));
        val.y = op(val.y, warpShflDown(val.y, offset));
        val.z = op(val.z, warpShflDown(val.z, offset));
    }
    return val;
}

template<typename Operation>
__device__ inline  int2 warpReduce(int2 val, Operation op)
{
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val.x = op(val.x, warpShflDown(val.x, offset));
        val.y = op(val.y, warpShflDown(val.y, offset));
    }
    return val;
}

template<typename Operation>
__device__ inline  int warpReduce(int val, Operation op)
{
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val = op(val, warpShflDown(val, offset));
    }
    return val;
}

//=======================================================================================
// per warp prefix sum
//=======================================================================================

template <typename T>
__device__ inline T warpInclusiveScan(T val) {
    int tid;
    tid = threadIdx.x % warpSize;
    for (int L = 1; L < warpSize; L <<= 1)
        val += (tid >= L) * warpShflUp(val, L);
    return val;
}

template <typename T>
__device__ inline T warpExclusiveScan(T val) {
    return warpInclusiveScan(val) - val;
}


//=======================================================================================
// Atomic functions
//=======================================================================================

// For `int64_t` which apparently maps to `long`.
__device__ inline long atomicAdd(long* address, long val)
{
    // Replacing with a supported function:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd
    using ull = unsigned long long;
    static_assert(sizeof(long) == sizeof(ull) || sizeof(long) == sizeof(int),
                  "No replacement found for `long atomicAdd(long*, long)`?!");

    if (sizeof(long) == sizeof(unsigned long long)) {
        return (long)atomicAdd((ull*)address, (ull)val);
    } else if (sizeof(long) == sizeof(int)) {
        return (long)atomicAdd((int*)address, (int)val);
    } else {
        return val;  // Unreachable.
    }
}

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

__device__ inline auto laneId() {return threadIdx.x % warpSize;}

// warning: warp id within one smx
__device__ inline uint32_t __warpid()
{
    uint32_t warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
    return warpid;
}

// warning: slower than threadIdx % warpSize
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
    return threadIdx.x & (warpSize-1);
}

template<>
__device__ inline uint getLaneId<2>()
{
    return ((threadIdx.y * blockDim.x) + threadIdx.x) & (warpSize-1);
}

template<>
__device__ inline uint getLaneId<3>()
{
    return (threadIdx.z * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) & (warpSize-1);
}

#if __CUDA_ARCH__ < 700

template<int DIMS=1>
__device__ inline int atomicAggInc(int *ctr)
{
    int lane_id = getLaneId<DIMS>();

    int mask = warpBallot(1);
    // select the leader
    int leader = __ffs(mask) - 1;
    // leader does the update
    int res;
    if(lane_id == leader)
    res = atomicAdd(ctr, __popc(mask));
    // broadcast result
    res = warpShfl(res, leader);
    // each thread computes its own value
    return res + __popc(mask & ((1 << lane_id) - 1));
}

#else

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ inline int atomicAggInc(int *ptr)
{
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;

    // elect the first active thread to perform atomic add
    if (g.thread_rank() == 0) {
        prev = atomicAdd(ptr, g.size());
    }

    // broadcast previous value within the warp
    // and add each active thread’s rank to it
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}

#endif

#else

inline float4 readNoCache(const float4* addr)
{
    return *addr;
}

#endif


__HD__ inline float fastPower(const float x, const float k)
{
    if (math::abs(k - 1.0f)   < 1e-6f) return x;
    if (math::abs(k - 0.5f)   < 1e-6f) return math::sqrt(math::abs(x));
    if (math::abs(k - 0.25f)  < 1e-6f) return math::sqrt(math::sqrt(math::abs(x)));

    return powf(math::abs(x), k);
}



