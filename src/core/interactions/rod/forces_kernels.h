#pragma once

#include "real.h"

#include <core/pvs/rod_vector.h>
#include <core/pvs/views/rv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>

namespace RodForcesKernels
{

struct GPU_RodBoundsParameters
{
    float lcenter, lcross, ldiag, lring;
    float kbounds;
};

__device__ inline real3 fetchPosition(const RVview& view, int i)
{
    Float3_int ri {view.readPosition(i)};
    return make_real3(ri.v);
}

// force exerted from r1 to r0
__device__ inline real3 fbound(const real3& r0, const real3& r1, const real& l0, const real& k)
{
    auto dr = r1 - r0;
    auto l = length(dr);
    return (k * (l - l0) / l) * dr;
}

__global__ void computeRodBoundForces(RVview view, GPU_RodBoundsParameters params)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int rodId     = i / view.nSegments;
    int segmentId = i % view.nSegments;
    int start = view.objSize * rodId + segmentId * 5;

    if (rodId     > view.nObjects ) return;
    if (segmentId > view.nSegments) return;

    auto r0 = fetchPosition(view, start + 0);
    auto u0 = fetchPosition(view, start + 1);
    auto u1 = fetchPosition(view, start + 2);
    auto v0 = fetchPosition(view, start + 3);
    auto v1 = fetchPosition(view, start + 4);
    auto r1 = fetchPosition(view, start + 5);

    real3 fr0{0._r}, fr1{0._r}, fu0{0._r}, fu1{0._r}, fv0{0._r}, fv1{0._r};

#define BOUND(a, b, l, k) do {                          \
        auto f = fbound(a, b, params. l, params. k);    \
        f##a += f;                                      \
        f##b -= f;                                      \
    } while(0)

    BOUND(r0, u0, ldiag, kbounds);
    BOUND(r0, u1, ldiag, kbounds);
    BOUND(r0, v0, ldiag, kbounds);
    BOUND(r0, v1, ldiag, kbounds);

    BOUND(r1, u0, ldiag, kbounds);
    BOUND(r1, u1, ldiag, kbounds);
    BOUND(r1, v0, ldiag, kbounds);
    BOUND(r1, v1, ldiag, kbounds);

    BOUND(u0, v0, lring, kbounds);
    BOUND(v0, u1, lring, kbounds);
    BOUND(u1, v1, lring, kbounds);
    BOUND(v1, u0, lring, kbounds);

    BOUND(u0, u1, lcross, kbounds);
    BOUND(v0, v1, lcross, kbounds);

    BOUND(r0, r1, lcenter, kbounds);

#undef BOUND
    
    atomicAdd(view.forces + start + 0, make_float3(fr0));
    atomicAdd(view.forces + start + 1, make_float3(fu0));
    atomicAdd(view.forces + start + 2, make_float3(fu1));
    atomicAdd(view.forces + start + 3, make_float3(fv0));
    atomicAdd(view.forces + start + 4, make_float3(fv1));
    atomicAdd(view.forces + start + 5, make_float3(fr1));
}

} // namespace RodForcesKernels
