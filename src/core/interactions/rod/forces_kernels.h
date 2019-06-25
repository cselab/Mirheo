#pragma once

#include "real.h"
#include "bisegment.h"

#include <core/pvs/rod_vector.h>
#include <core/pvs/views/rv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>

struct GPU_RodBoundsParameters
{
    float lcenter, lcross, ldiag, lring;
    float ksCenter, ksFrame;
};

struct GPU_SpinParameters
{
    float J, kBT, beta, seed;
};


namespace RodForcesKernels
{

// elastic force exerted from p1 to p0
__device__ inline real3 fbound(const real3& r0, const real3& r1, const float ks, float l0)
{
    auto dr = r1 - r0;
    auto l = length(dr);
    auto xi = (l - l0);
    auto linv = 1.0_r / l;

    auto fmagn = ks * xi * (0.5_r * xi + l);
    
    return (linv * fmagn) * dr;
}

__global__ void computeRodBoundForces(RVview view, GPU_RodBoundsParameters params)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int rodId     = i / view.nSegments;
    const int segmentId = i % view.nSegments;
    const int start = view.objSize * rodId + segmentId * 5;

    if (rodId     >= view.nObjects ) return;
    if (segmentId >= view.nSegments) return;

    auto r0 = fetchPosition(view, start + 0);
    auto u0 = fetchPosition(view, start + 1);
    auto u1 = fetchPosition(view, start + 2);
    auto v0 = fetchPosition(view, start + 3);
    auto v1 = fetchPosition(view, start + 4);
    auto r1 = fetchPosition(view, start + 5);

    real3 fr0{0._r, 0._r, 0._r}, fr1{0._r, 0._r, 0._r};
    real3 fu0{0._r, 0._r, 0._r}, fu1{0._r, 0._r, 0._r};
    real3 fv0{0._r, 0._r, 0._r}, fv1{0._r, 0._r, 0._r};

#define BOUND(a, b, k, l) do {                          \
        auto f = fbound(a, b, params. k, params. l);    \
        f##a += f;                                      \
        f##b -= f;                                      \
    } while(0)

    BOUND(r0, u0, ksFrame, ldiag);
    BOUND(r0, u1, ksFrame, ldiag);
    BOUND(r0, v0, ksFrame, ldiag);
    BOUND(r0, v1, ksFrame, ldiag);

    BOUND(r1, u0, ksFrame, ldiag);
    BOUND(r1, u1, ksFrame, ldiag);
    BOUND(r1, v0, ksFrame, ldiag);
    BOUND(r1, v1, ksFrame, ldiag);

    BOUND(u0, v0, ksFrame, lring);
    BOUND(v0, u1, ksFrame, lring);
    BOUND(u1, v1, ksFrame, lring);
    BOUND(v1, u0, ksFrame, lring);

    BOUND(u0, u1, ksFrame, lcross);
    BOUND(v0, v1, ksFrame, lcross);

    BOUND(r0, r1, ksCenter, lcenter);

#undef BOUND
    
    atomicAdd(view.forces + start + 0, make_float3(fr0));
    atomicAdd(view.forces + start + 1, make_float3(fu0));
    atomicAdd(view.forces + start + 2, make_float3(fu1));
    atomicAdd(view.forces + start + 3, make_float3(fv0));
    atomicAdd(view.forces + start + 4, make_float3(fv1));
    atomicAdd(view.forces + start + 5, make_float3(fr1));
}


__device__ inline void writeBisegmentData(int i, float4 *kappa, float2 *tau_l,
                                          real2 k0, real2 k1, real tau, real l)
{
    kappa[i] = { (float) k0.x, (float) k0.y,
                 (float) k1.x, (float) k1.y };

    tau_l[i] = { (float) tau, (float) l };
}

__device__ inline void fetchBisegmentData(int i, const float4 *kappa, const float2 *tau_l,
                                          real2& k0, real2& k1, real& tau, real& l)
{
    auto ks = kappa[i];
    auto tl = tau_l[i];
    k0.x = ks.x;
    k0.y = ks.y;

    k1.x = ks.z;
    k1.y = ks.w;

    tau = tl.x;
    l = tl.y;
}

__global__ void computeBisegmentData(RVview view, float4 *kappa, float2 *tau_l)
{
    constexpr int stride = 5;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int nBiSegments = view.nSegments - 1;
    const int rodId       = i / nBiSegments;
    const int biSegmentId = i % nBiSegments;
    const int start = view.objSize * rodId + biSegmentId * stride;

    if (rodId       >= view.nObjects ) return;
    if (biSegmentId >= nBiSegments   ) return;

    const BiSegment<0> bisegment(view, start);

    real2 k0, k1;
    real tau;
    bisegment.computeCurvatures(k0, k1);
    bisegment.computeTorsion(tau);

    writeBisegmentData(i, kappa, tau_l, k0, k1, tau, bisegment.l);
}

template <int Nstates>
__global__ void findPolymorphicStates(RVview view, GPU_RodBiSegmentParameters<Nstates> params, const float4 *kappa, const float2 *tau_l)
{
    const int tid   = threadIdx.x;
    const int rodId = blockIdx.x;
    
    const int nBiSegments = view.nSegments - 1;

    for (int biSegmentId = tid; biSegmentId < nBiSegments; ++biSegmentId)
    {
        real2 k0, k1;
        real tau, l;
        int i = rodId * nBiSegments + biSegmentId;

        fetchBisegmentData(i, kappa, tau_l, k0, k1, tau, l);

        int state = 0;
        real E = computeEnergy(l, k0, k1, tau, state, params);            
        
        #pragma unroll
        for (int s = 1; s < Nstates; ++s)
        {
            real Es = computeEnergy(l, k0, k1, tau, s, params);
            if (Es < E)
            {
                E = Es;
                state = s;
            }
        }

        view.states[i] = state;
    }
}

template <int Nstates>
__device__ inline int randomOtherState(int current, float seed)
{
    float u = Saru::mean0var1(seed, threadIdx.x, 123456 * current + 98765 * blockIdx.x );
    int s = (Nstates - 1) * u;
    return s >= current ? s + 1 : s;
}

template <int Nstates>
__device__ inline int acceptReject(int sprev, int scurrent, int snext,
                                   const real2& k0, const real2& k1, const real& tau, const real& l,
                                   const GPU_RodBiSegmentParameters<Nstates>& params,
                                   const GPU_SpinParameters& spinParams)
{
    int sother = randomOtherState<Nstates>(scurrent, spinParams.seed);

    float Ecurrent = computeEnergy(l, k0, k1, tau, scurrent, params)
        + spinParams.J * (abs(scurrent-sprev) + abs(scurrent-snext));

    float Eother   = computeEnergy(l, k0, k1, tau, sother  , params)
        + spinParams.J * (abs(sother  -sprev) + abs(sother  -snext));

    float dE = Eother - Ecurrent;
    
    float u = Saru::mean0var1(spinParams.seed, 12345 * threadIdx.x - 6789, 123456 * sother + 98765 * blockIdx.x );

    if (spinParams.kBT < 1e-6)
        return dE > 0 ? scurrent : sother;
    
    if (u < exp(-dE * spinParams.beta))
        return sother;

    return scurrent;
}

template <int Nstates>
__global__ void findPolymorphicStatesMCStep(RVview view, GPU_RodBiSegmentParameters<Nstates> params,
                                            GPU_SpinParameters spinParams, const float4 *kappa, const float2 *tau_l)
{
    const int tid   = threadIdx.x;
    const int rodId = blockIdx.x;
    
    const int nBiSegments = view.nSegments - 1;

    extern __shared__ int *states;

    for (int biSegmentId = tid; biSegmentId < nBiSegments; ++biSegmentId)
    {
        int i = rodId * nBiSegments + biSegmentId;
        states[biSegmentId] = view.states[i];
    }

    __syncthreads();

    auto execPhase = [&](int odd)
    {
        for (int biSegmentId = tid; biSegmentId < nBiSegments; ++biSegmentId)
        {
            if (biSegmentId % 2 == odd) continue;
            
            real2 k0, k1;
            real tau, l;
            int i = rodId * nBiSegments + biSegmentId;
            
            fetchBisegmentData(i, kappa, tau_l, k0, k1, tau, l);

            int scurrent = states[biSegmentId];
            int sprev = states[max(biSegmentId - 1, 0          )];
            int snext = states[min(biSegmentId + 1, nBiSegments)];

            states[biSegmentId] = acceptReject(sprev, scurrent, snext, k0, k1, tau, l, params, spinParams);
        }
    };

    execPhase(0);
    __syncthreads();

    execPhase(1);
    __syncthreads();

    for (int biSegmentId = tid; biSegmentId < nBiSegments; ++biSegmentId)
    {
        int i = rodId * nBiSegments + biSegmentId;
        view.states[i] = states[biSegmentId];
    }
}

template <int Nstates>
__device__ inline int getState(const RVview& view, int i)
{
    if (Nstates > 1) return view.states[i];
    else             return 0;
}

template <int Nstates>
__global__ void computeRodBiSegmentForces(RVview view, GPU_RodBiSegmentParameters<Nstates> params, bool saveEnergies)
{
    constexpr int stride = 5;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int nBiSegments = view.nSegments - 1;
    const int rodId       = i / nBiSegments;
    const int biSegmentId = i % nBiSegments;
    const int start = view.objSize * rodId + biSegmentId * stride;

    if (rodId       >= view.nObjects ) return;
    if (biSegmentId >= nBiSegments   ) return;

    const BiSegment<Nstates> bisegment(view, start);

    real3 fr0, fr2, fpm0, fpm1;
    fr0 = fr2 = fpm0 = fpm1 = make_real3(0.0_r);

    const int state = getState<Nstates>(view, i);
    
    bisegment.computeBendingForces(state, params, fr0, fr2, fpm0, fpm1);
    bisegment.computeTwistForces  (state, params, fr0, fr2, fpm0, fpm1);

    // by conservation of momentum
    auto fr1  = -(fr0 + fr2);
    auto fpp0 = -fpm0;
    auto fpp1 = -fpm1;
    
    atomicAdd(view.forces + start + 0 * stride, make_float3(fr0));
    atomicAdd(view.forces + start + 1 * stride, make_float3(fr1));
    atomicAdd(view.forces + start + 2 * stride, make_float3(fr2));

    atomicAdd(view.forces + start +          1, make_float3(fpm0));
    atomicAdd(view.forces + start +          2, make_float3(fpp0));
    atomicAdd(view.forces + start + stride + 1, make_float3(fpm1));
    atomicAdd(view.forces + start + stride + 2, make_float3(fpp1));

    if (saveEnergies) view.energies[i] = bisegment.computeEnergy(state, params);
}


} // namespace RodForcesKernels
