#pragma once

#include "kernels/real.h"
#include "kernels/bisegment.h"

#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/views/rv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>

namespace mirheo
{

/// device-compatible structure for the ising model of polymorphic states transition
struct GPU_SpinParameters
{
    real J;    ///< Ising energy coupling
    real kBT;  ///< temperature in energy units
    real beta; ///< 1/kBT
    real seed; ///< random seed
};

namespace rod_states_kernels
{

__device__ inline void writeBisegmentData(int i, real4 *kappa, real2 *tau_l,
                                          rReal2 k0, rReal2 k1, rReal tau, rReal l)
{
    kappa[i] = { (real) k0.x, (real) k0.y,
                 (real) k1.x, (real) k1.y };

    tau_l[i] = { (real) tau, (real) l };
}

__device__ inline void fetchBisegmentData(int i, const real4 *kappa, const real2 *tau_l,
                                          rReal2& k0, rReal2& k1, rReal& tau, rReal& l)
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

__global__ void computeBisegmentData(RVview view, real4 *kappa, real2 *tau_l)
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

    rReal2 k0, k1;
    rReal tau;
    bisegment.computeCurvatures(k0, k1);
    bisegment.computeTorsion(tau);

    writeBisegmentData(i, kappa, tau_l, k0, k1, tau, bisegment.l);
}

template <int Nstates>
__global__ void findPolymorphicStates(RVview view, GPU_RodBiSegmentParameters<Nstates> params, const real4 *kappa, const real2 *tau_l)
{
    const int tid   = threadIdx.x;
    const int rodId = blockIdx.x;
    
    const int nBiSegments = view.nSegments - 1;

    for (int biSegmentId = tid; biSegmentId < nBiSegments; ++biSegmentId)
    {
        rReal2 k0, k1;
        rReal tau, l;
        int i = rodId * nBiSegments + biSegmentId;

        fetchBisegmentData(i, kappa, tau_l, k0, k1, tau, l);

        int state = 0;
        rReal E = computeEnergy(l, k0, k1, tau, state, params);            
        
        #pragma unroll
        for (int s = 1; s < Nstates; ++s)
        {
            rReal Es = computeEnergy(l, k0, k1, tau, s, params);
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
__device__ inline int randomOtherState(int current, real seed)
{
    real u = Saru::uniform01(seed, threadIdx.x, 123456 * current + 98765 * blockIdx.x );
    unsigned int r = static_cast<unsigned int>(4294967295.0_r * u);
    int s = r % (Nstates - 1);
    return s >= current ? (s + 1) % Nstates : s;
}

template <int Nstates>
__device__ inline int acceptReject(int sprev, int scurrent, int snext,
                                   const rReal2& k0, const rReal2& k1, const rReal& tau, const rReal& l,
                                   const GPU_RodBiSegmentParameters<Nstates>& params,
                                   const GPU_SpinParameters& spinParams)
{
    int sother = randomOtherState<Nstates>(scurrent, spinParams.seed);

    real Ecurrent = computeEnergy(l, k0, k1, tau, scurrent, params)
        + spinParams.J * (math::abs(scurrent-sprev) + math::abs(scurrent-snext));

    real Eother   = computeEnergy(l, k0, k1, tau, sother  , params)
        + spinParams.J * (math::abs(sother  -sprev) + math::abs(sother  -snext));

    real dE = Eother - Ecurrent;
    
    real u = Saru::uniform01(spinParams.seed, 12345 * threadIdx.x - 6789, 123456 * sother + 98765 * blockIdx.x );

    if (spinParams.kBT < 1e-6)
        return dE < 0 ? sother : scurrent;
    
    if (u < math::exp(-dE * spinParams.beta))
        return sother;

    return scurrent;
}

template <int Nstates>
__global__ void findPolymorphicStatesMCStep(RVview view, GPU_RodBiSegmentParameters<Nstates> params,
                                            GPU_SpinParameters spinParams, const real4 *kappa, const real2 *tau_l)
{
    const int tid   = threadIdx.x;
    const int rodId = blockIdx.x;
    
    const int nBiSegments = view.nSegments - 1;

    extern __shared__ int states[];

    for (int biSegmentId = tid; biSegmentId < nBiSegments; biSegmentId += blockDim.x)
    {
        int i = rodId * nBiSegments + biSegmentId;
        states[biSegmentId] = view.states[i];
    }

    __syncthreads();

    auto execPhase = [&](int evenOdd)
    {
        for (int biSegmentId = tid; biSegmentId < nBiSegments; biSegmentId += blockDim.x)
        {
            if (biSegmentId % 2 == evenOdd) continue;
            
            rReal2 k0, k1;
            rReal tau, l;
            int i = rodId * nBiSegments + biSegmentId;
            
            fetchBisegmentData(i, kappa, tau_l, k0, k1, tau, l);

            int scurrent = states[biSegmentId];
            int sprev = states[math::max(biSegmentId - 1, 0            )];
            int snext = states[math::min(biSegmentId + 1, nBiSegments-1)];

            states[biSegmentId] = acceptReject(sprev, scurrent, snext,
                                               k0, k1, tau, l, params, spinParams);
        }
    };

    constexpr int evenStep = 0;
    constexpr int  oddStep = 1;
    
    execPhase(evenStep);
    __syncthreads();

    execPhase(oddStep);
    __syncthreads();

    for (int biSegmentId = tid; biSegmentId < nBiSegments; biSegmentId += blockDim.x)
    {
        int i = rodId * nBiSegments + biSegmentId;
        view.states[i] = states[biSegmentId];
    }
}

} // namespace rod_states_kernels

} // namespace mirheo
