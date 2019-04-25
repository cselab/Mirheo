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
    float kBounds, kVisc;
};

struct GPU_RodBiSegmentParameters
{
    float3 kBending;
    float2 omegaEq;
    float kTwist, tauEq;
};


// force exerted from p1 to p0
__device__ inline real3 fbound(const ParticleReal& p0, const ParticleReal& p1,
                               const GPU_RodBoundsParameters& params, float l0)
{
    auto dr = p1.r - p0.r;
    auto du = p1.u - p0.u;
    auto l = length(dr);
    auto linv = 1.0_r / l;

    auto fMagnElastic = params.kBounds * (l - l0);
    auto fMagnViscous = params.kVisc   * dot(dr, du) * linv;

    return (linv * (fMagnElastic + fMagnViscous)) * dr;
}

__global__ void computeRodBoundForces(RVview view, GPU_RodBoundsParameters params)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int rodId     = i / view.nSegments;
    const int segmentId = i % view.nSegments;
    const int start = view.objSize * rodId + segmentId * 5;

    if (rodId     >= view.nObjects ) return;
    if (segmentId >= view.nSegments) return;

    auto r0 = fetchParticle(view, start + 0);
    auto u0 = fetchParticle(view, start + 1);
    auto u1 = fetchParticle(view, start + 2);
    auto v0 = fetchParticle(view, start + 3);
    auto v1 = fetchParticle(view, start + 4);
    auto r1 = fetchParticle(view, start + 5);

    real3 fr0{0._r, 0._r, 0._r}, fr1{0._r, 0._r, 0._r};
    real3 fu0{0._r, 0._r, 0._r}, fu1{0._r, 0._r, 0._r};
    real3 fv0{0._r, 0._r, 0._r}, fv1{0._r, 0._r, 0._r};

#define BOUND(a, b, l) do {                          \
        auto f = fbound(a, b, params, params. l);       \
        f##a += f;                                      \
        f##b -= f;                                      \
    } while(0)

    BOUND(r0, u0, ldiag);
    BOUND(r0, u1, ldiag);
    BOUND(r0, v0, ldiag);
    BOUND(r0, v1, ldiag);

    BOUND(r1, u0, ldiag);
    BOUND(r1, u1, ldiag);
    BOUND(r1, v0, ldiag);
    BOUND(r1, v1, ldiag);

    BOUND(u0, v0, lring);
    BOUND(v0, u1, lring);
    BOUND(u1, v1, lring);
    BOUND(v1, u0, lring);

    BOUND(u0, u1, lcross);
    BOUND(v0, v1, lcross);

    BOUND(r0, r1, lcenter);

#undef BOUND
    
    atomicAdd(view.forces + start + 0, make_float3(fr0));
    atomicAdd(view.forces + start + 1, make_float3(fu0));
    atomicAdd(view.forces + start + 2, make_float3(fu1));
    atomicAdd(view.forces + start + 3, make_float3(fv0));
    atomicAdd(view.forces + start + 4, make_float3(fv1));
    atomicAdd(view.forces + start + 5, make_float3(fr1));
}


__device__ inline real3 fetchBishopFrame(const RVview& view, int objId, int segmentId)
{
    float3 u =  view.bishopFrames[objId * view.nSegments + segmentId];
    return make_real3(u);
}

// theta0 and theta1 might be close to pi, leading to +- pi values
// this function compute the difference between the angles such as it is safely less that pi
__device__ inline real safeDiffTheta(real t0, real t1)
{
    auto dth = t1 - t0;
    if (dth >  M_PI) dth -= 2.0_r * M_PI;
    if (dth < -M_PI) dth += 2.0_r * M_PI;
    return dth;
}

__device__ inline real2 symmetricMatMult(const real3& A, const real2& x)
{
    return {A.x * x.x + A.y * x.y,
            A.y * x.x + A.z * x.y};
}

struct BiSegment
{
    real3 e0, e1, t0, t1, dp0, dp1, bicur;
    real bicurFactor, e0inv, e1inv, linv;

    __device__ inline BiSegment(const RVview& view, int start)
    {
        auto r0  = fetchPosition(view, start + 0);
        auto r1  = fetchPosition(view, start + 5);
        auto r2  = fetchPosition(view, start + 10);
        
        auto pm0 = fetchPosition(view, start + 1);
        auto pp0 = fetchPosition(view, start + 2);
        auto pm1 = fetchPosition(view, start + 6);
        auto pp1 = fetchPosition(view, start + 7);
        
        e0 = r1 - r0;
        e1 = r2 - r1;

        t0 = normalize(e0);
        t1 = normalize(e1);

        dp0 = pp0 - pm0;
        dp1 = pp1 - pm1;

        real le0 = length(e0);
        real le1 = length(e1);
        e0inv = 1.0_r / le0;
        e1inv = 1.0_r / le1;
        linv = 2.0_r / (le0 + le1);
        bicurFactor = 1.0_r / (le0 * le1 + dot(e0, e1));
        bicur = (2.0_r * bicurFactor) * cross(e0, e1);
    }

    __device__ inline real3 applyGrad0Bicur(const real3& v) const
    {
        return bicurFactor * (2.0_r * cross(e0, v) + dot(e0, v) * bicur);
    }

    __device__ inline real3 applyGrad2Bicur(const real3& v) const
    {
        return bicurFactor * (2.0_r * cross(e1, v) + dot(e1, v) * bicur);
    }

    __device__ inline void computeBendingForces(const GPU_RodBiSegmentParameters& params,
                                                real3& fr0, real3& fr2, real3& fpm0, real3& fpm1) const
    {
        real dpt0 = dot(dp0, t0);
        real dpt1 = dot(dp1, t1);

        real3 t0_dp0 = cross(t0, dp0);
        real3 t1_dp1 = cross(t1, dp1);
    
        real3 dpPerp0 = dp0 - dpt0 * t0;
        real3 dpPerp1 = dp1 - dpt1 * t1;

        real dpPerp0inv = rsqrt(dot(dpPerp0, dpPerp0));
        real dpPerp1inv = rsqrt(dot(dpPerp1, dpPerp1));
    
        real2 omega0 { +dpPerp0inv * dot(bicur, t0_dp0),
                       -dpPerp0inv * dot(bicur,    dp0)};

        real2 omega1 { +dpPerp1inv * dot(bicur, t1_dp1),
                       -dpPerp1inv * dot(bicur,    dp1)};

        real2 domega0 = omega0 - make_real2(params.omegaEq);
        real2 domega1 = omega1 - make_real2(params.omegaEq);

        real2 Bomega0 = symmetricMatMult(make_real3(params.kBending), domega0);
        real2 Bomega1 = symmetricMatMult(make_real3(params.kBending), domega1);

        real Eb = 0.5_r * linv * (dot(domega0, Bomega0) + dot(domega1, Bomega1));

        real3 grad0NormOmega0 = (-e0inv * dpPerp0inv * dpPerp0inv * dpt0) * dpPerp0;
        real3 grad2NormOmega1 = ( e1inv * dpPerp1inv * dpPerp1inv * dpt1) * dpPerp1;

        // 1. contributions of center line:    
        real3 baseGradOmega0x = cross(bicur, dp0) + dot(bicur, t0_dp0) * t0;
        real3 baseGradOmega1x = cross(bicur, dp1) + dot(bicur, t1_dp1) * t1;
    

        real3 grad0Omega0x = omega0.x * grad0NormOmega0 + dpPerp0inv * (applyGrad0Bicur(t0_dp0) - e0inv * baseGradOmega0x);
        real3 grad2Omega0x =                              dpPerp0inv *  applyGrad2Bicur(t0_dp0);

        real3 grad0Omega0y = omega0.y * grad0NormOmega0 - dpPerp0inv *  applyGrad0Bicur(   dp0);
        real3 grad2Omega0y =                            - dpPerp0inv *  applyGrad2Bicur(   dp0);

        real3 grad2Omega1x = omega1.x * grad2NormOmega1 + dpPerp1inv * (applyGrad2Bicur(t1_dp1) + e1inv * baseGradOmega1x);
        real3 grad0Omega1x =                              dpPerp1inv *  applyGrad0Bicur(t1_dp1);

        real3 grad2Omega1y = omega1.y * grad2NormOmega1 - dpPerp1inv *  applyGrad2Bicur(   dp1);
        real3 grad0Omega1y =                            - dpPerp1inv *  applyGrad0Bicur(   dp1);
    
        // 1.a contribution of omega
        fr0 += linv * (Bomega0.x * grad0Omega0x + Bomega0.y * grad0Omega0y  +  Bomega1.x * grad0Omega1x + Bomega1.y * grad0Omega1y);
        fr2 += linv * (Bomega0.x * grad2Omega0x + Bomega0.y * grad2Omega0y  +  Bomega1.x * grad2Omega1x + Bomega1.y * grad2Omega1y);

        // 1.b contribution of l
        fr0 += (-0.5_r * linv * Eb) * t0;
        fr2 += ( 0.5_r * linv * Eb) * t1;


        // 2. contributions material frame:

        real3 baseGradOmegaMF0 = (- dpPerp0inv * dpPerp0inv) * dpPerp0;
        real3 baseGradOmegaMF1 = (- dpPerp1inv * dpPerp1inv) * dpPerp1;

        real3 gradOmegaMF0x = omega0.x * baseGradOmegaMF0 + dpPerp0inv * cross(bicur, t0);
        real3 gradOmegaMF0y = omega0.y * baseGradOmegaMF0 - dpPerp0inv * bicur;

        real3 gradOmegaMF1x = omega1.x * baseGradOmegaMF1 + dpPerp1inv * cross(bicur, t1);
        real3 gradOmegaMF1y = omega1.y * baseGradOmegaMF1 - dpPerp1inv * bicur;

        fpm0 += linv * (Bomega0.x * gradOmegaMF0x + Bomega0.y * gradOmegaMF0y);
        fpm1 += linv * (Bomega1.x * gradOmegaMF1x + Bomega1.y * gradOmegaMF1y);
    }

    __device__ inline void computeTwistForces(const GPU_RodBiSegmentParameters& params,
                                              const real3& u0, const real3& u1,
                                              real3& fr0, real3& fr2, real3& fpm0, real3& fpm1) const
    {
        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        real dpu0 = dot(dp0, u0);
        real dpv0 = dot(dp0, v0);

        real dpu1 = dot(dp1, u1);
        real dpv1 = dot(dp1, v1);

        real theta0 = atan2(dpv0, dpu0);
        real theta1 = atan2(dpv1, dpu1);
    
        real dtheta_l = safeDiffTheta(theta0, theta1) * linv;
        real dtheta_l_mtau = dtheta_l - params.tauEq;
        real dtheta_l_ptau = dtheta_l + params.tauEq;

        real ftwistLFactor = params.kTwist * dtheta_l_ptau * dtheta_l_mtau;

        fr0 -= 0.5_r * ftwistLFactor * t0;
        fr2 += 0.5_r * ftwistLFactor * t1;

        real dthetaFFactor = 2.0_r * dtheta_l_mtau * params.kTwist;

        fr0 += (0.5_r * dthetaFFactor * e0inv) * bicur;
        fr2 -= (0.5_r * dthetaFFactor * e1inv) * bicur;

        fpm0 += (dthetaFFactor / (dpu0*dpu0 + dpv0*dpv0)) * (dpv0 * u0 - dpu0 * v0);
        fpm1 += (dthetaFFactor / (dpu1*dpu1 + dpv1*dpv1)) * (dpu1 * v1 - dpv1 * u1);
    }
};

__global__ void computeRodBiSegmentForces(RVview view, GPU_RodBiSegmentParameters params)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int nBiSegments = view.nSegments - 1;
    const int rodId       = i / nBiSegments;
    const int biSegmentId = i % nBiSegments;
    const int start = view.objSize * rodId + biSegmentId * 5;

    if (rodId       >= view.nObjects ) return;
    if (biSegmentId >= nBiSegments   ) return;

    const BiSegment bisegment(view, start);

    real3 fr0, fr2, fpm0, fpm1;
    fr0 = fr2 = fpm0 = fpm1 = make_real3(0.0_r);
    
    bisegment.computeBendingForces(params, fr0, fr2, fpm0, fpm1);
    
    auto u0 = fetchBishopFrame(view, rodId, biSegmentId + 0);
    auto u1 = fetchBishopFrame(view, rodId, biSegmentId + 1);

    bisegment.computeTwistForces(params, u0, u1, fr0, fr2, fpm0, fpm1);

    // by conservation of momentum
    auto fr1  = -(fr0 + fr2);
    auto fpp0 = -fpm0;
    auto fpp1 = -fpm1;
    
    atomicAdd(view.forces + start +  0, make_float3(fr0));
    atomicAdd(view.forces + start +  5, make_float3(fr1));
    atomicAdd(view.forces + start + 10, make_float3(fr2));

    atomicAdd(view.forces + start +  1, make_float3(fpm0));
    atomicAdd(view.forces + start +  2, make_float3(fpp0));
    atomicAdd(view.forces + start +  6, make_float3(fpm1));
    atomicAdd(view.forces + start +  7, make_float3(fpp1));
}


} // namespace RodForcesKernels
