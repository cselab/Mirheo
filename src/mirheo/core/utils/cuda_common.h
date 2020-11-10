// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "helper_math.h"
#include "cpu_gpu_defines.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#include <cooperative_groups.h>
#endif

namespace mirheo
{

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

/** \brief compute the number of blocks for a given problem size and block size
    \param [in] n Problem size
    \param [in] nthreads Block size
    \return Number of required blocks
 */
inline int getNblocks(const int n, const int nthreads)
{
    return (n+nthreads-1) / nthreads;
}

/// \return square of the input value
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

/** \brief Perform a reduction on a full warp
    \param val The value to reduce
    \param op The operation to operate between the input data (e.g. addition). Must be associative.
    \return The reduced value. Only available on laneId == 0

    This function Must be called by all threads in the warp.
 */
template<typename Operation>
__device__ inline  float4 warpReduce(float4 val, Operation op)
{
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val.x = op(val.x, warpShflDown(val.x, offset));
        val.y = op(val.y, warpShflDown(val.y, offset));
        val.z = op(val.z, warpShflDown(val.z, offset));
        val.w = op(val.w, warpShflDown(val.w, offset));
    }
    return val;
}

/// See warpReduce()
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

/// See warpReduce()
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

/// See warpReduce()
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

/// See warpReduce()
template<typename Operation>
__device__ inline  double4 warpReduce(double4 val, Operation op)
{
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val.x = op(val.x, warpShflDown(val.x, offset));
        val.y = op(val.y, warpShflDown(val.y, offset));
        val.z = op(val.z, warpShflDown(val.z, offset));
        val.w = op(val.w, warpShflDown(val.w, offset));
    }
    return val;
}

/// See warpReduce()
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

/// See warpReduce()
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

/// See warpReduce()
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

/// See warpReduce()
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

/// See warpReduce()
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

/// See warpReduce()
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


/** \brief Perform an inclusive prefix sum on a full warp
    \param val The value to reduce (one per lane index)
    \return The prefix sum at the given lane index

    This function Must be called by all threads in the warp.
 */
template <typename T>
__device__ inline T warpInclusiveScan(T val) {
    int tid;
    tid = threadIdx.x % warpSize;
    for (int L = 1; L < warpSize; L <<= 1)
        val += (tid >= L) * warpShflUp(val, L);
    return val;
}

/** \brief Perform an exclusive prefix sum on a full warp
    \param val The value to reduce (one per lane index)
    \return The prefix sum at the given lane index

    This function Must be called by all threads in the warp.
 */
template <typename T>
__device__ inline T warpExclusiveScan(T val) {
    return warpInclusiveScan(val) - val;
}

} // namespace mirheo

//=======================================================================================
// Atomic functions
//=======================================================================================

/// Overload atomicAdd for `int64_t` which apparently maps to `long`.
__device__ inline long atomicAdd(long *address, long val)
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
/// Overload atomicAdd for `double` in old cuda versions
__device__ inline double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
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

__device__ inline float4 atomicAdd(float4* addr, float4 v)
{
    float4 res;
    res.x = atomicAdd((float*)addr,   v.x);
    res.y = atomicAdd((float*)addr+1, v.y);
    res.z = atomicAdd((float*)addr+2, v.z);
    res.w = atomicAdd((float*)addr+3, v.w);
    return res;
}


__device__ inline double2 atomicAdd(double2* addr, double2 v)
{
    double2 res;
    res.x = atomicAdd((double*)addr,   v.x);
    res.y = atomicAdd((double*)addr+1, v.y);
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

__device__ inline double3 atomicAdd(double4* addr, double3 v)
{
    double3 res;
    res.x = atomicAdd((double*)addr,   v.x);
    res.y = atomicAdd((double*)addr+1, v.y);
    res.z = atomicAdd((double*)addr+2, v.z);
    return res;
}

__device__ inline double4 atomicAdd(double4* addr, double4 v)
{
    double4 res;
    res.x = atomicAdd((double*)addr,   v.x);
    res.y = atomicAdd((double*)addr+1, v.y);
    res.z = atomicAdd((double*)addr+2, v.z);
    res.w = atomicAdd((double*)addr+3, v.w);
    return res;
}

namespace mirheo
{

//=======================================================================================
// Read/write through cache
//=======================================================================================

/// read a \c float4 value directly from global memory to reduce cache pressure on concurrent kernels
__device__ inline float4 readNoCache(const float4 *addr)
{
    float4 res;
    asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(res.x), "=f"(res.y), "=f"(res.z), "=f"(res.w) : "l"(addr));
    return res;
}

/// write a \c float4 value directly to global memory to reduce cache pressure on concurrent kernels
__device__ inline void writeNoCache(float4 *addr, const float4 val)
{
    asm("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));
}

/// write a \c double4 value directly from global memory to reduce cache pressure on concurrent kernels
__device__ inline double4 readNoCache(const double4 *addr)
{
    auto addr2 = reinterpret_cast<const double2*>(addr);
    double4 res;
    asm("ld.global.cv.v2.f64 {%0, %1}, [%2];" : "=d"(res.x), "=d"(res.y) : "l"(addr2 + 0));
    asm("ld.global.cv.v2.f64 {%0, %1}, [%2];" : "=d"(res.z), "=d"(res.w) : "l"(addr2 + 1));
    return res;
}

/// write a \c double4 value directly to global memory to reduce cache pressure on concurrent kernels
__device__ inline void writeNoCache(double4 *addr, const double4 val)
{
    auto addr2 = reinterpret_cast<double2*>(addr);
    asm("st.global.wt.v2.f64 [%0], {%1, %2};" :: "l"(addr2+0), "d"(val.x), "d"(val.y));
    asm("st.global.wt.v2.f64 [%0], {%1, %2};" :: "l"(addr2+1), "d"(val.z), "d"(val.w));
}



//=======================================================================================
// Lane and warp id
// https://stackoverflow.com/questions/28881491/how-can-i-find-out-which-thread-is-getting-executed-on-which-core-of-the-gpu
//=======================================================================================

/** \brief compute the lane index (index within a warp) of the current thread
    \return lane index

    \rst
    .. warning::
        This is only valid if the block size is in one dimension. See __laneId() in more dimensions.
    \endrst
 */
__device__ inline auto laneId() {return threadIdx.x % warpSize;}

/** \brief compute the warp index within the current SMX
    \return warp index

    \rst
    .. warning::
        This value will depend on the architecture.
        This will not be give unique warp index per warp within one kernel.
    \endrst
 */
__device__ inline uint32_t __warpid()
{
    uint32_t warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
    return warpid;
}

// warning: slower than threadIdx % warpSize
/** \brief compute the lane index (index within a warp) of the current thread.
    \return lane index

    This will give the correct lane id even in the multi dimensional block case, unlike laneId().
    However, it is slower than the latter in the linear (1D) case.
 */
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

/// helper function; return the lane Id of the current thread
template<int DIMS>
__device__ inline uint getLaneId();

/// helper function; return the lane Id of the current thread when block size is 1D
template<>
__device__ inline uint getLaneId<1>()
{
    return threadIdx.x & (warpSize-1);
}

/// helper function; return the lane Id of the current thread when block size is 2D
template<>
__device__ inline uint getLaneId<2>()
{
    return ((threadIdx.y * blockDim.x) + threadIdx.x) & (warpSize-1);
}

/// helper function; return the lane Id of the current thread when block size is 3D
template<>
__device__ inline uint getLaneId<3>()
{
    return (threadIdx.z * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) & (warpSize-1);
}

#if __CUDA_ARCH__ < 700

/** \brief warp aggregated atomics, used to reduce the number of atomic operations on a warp.
    \param ptr location of the value to increment atomically. Must be the same for all threads that call this function within a warp.
    \return the incremented value of *ptr that woul be returned by \c atomicAdd(ptr, 1).

    Not all threads within the warp need to call this function.
    This is equivalent to call \c atomicAdd(ptr, 1) by all the callers of this function.
 */
template<int DIMS=1>
__device__ inline int atomicAggInc(int *ptr)
{
    int lane_id = getLaneId<DIMS>();

    int mask = warpBallot(1);
    // select the leader
    int leader = __ffs(mask) - 1;
    // leader does the update
    int res;
    if(lane_id == leader)
        res = atomicAdd(ptr, __popc(mask));
    // broadcast result
    res = warpShfl(res, leader);
    // each thread computes its own value
    return res + __popc(mask & ((1 << lane_id) - 1));
}

#else

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

/// cpu compatible overload; just to make things compile
inline float4 readNoCache(const float4 *addr)
{
    return *addr;
}

/// cpu compatible overload; just to make things compile
inline double4 readNoCache(const double4 *addr)
{
    return *addr;
}

#endif

/** \brief Compute |x|**k
    \param x The value to take the power to
    \param k The exponent
    \return |x|**k

    This function may lead faster performance for k = 1, 0.5, 0.25 than pow.
 */
__HD__ inline float fastPower(const float x, const float k)
{
    constexpr real eps = 1e-6_r;
    if (math::abs(k - 1.0_r)   < eps) return x;
    if (math::abs(k - 0.5_r)   < eps) return math::sqrt(math::abs(x));
    if (math::abs(k - 0.25_r)  < eps) return math::sqrt(math::sqrt(math::abs(x)));

    return math::pow(math::abs(x), k);
}

} // namespace mirheo
