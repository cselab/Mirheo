#pragma once

#include "real.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/quaternion.h>
#include <core/pvs/views/rv.h>

template<int Nstates>
struct GPU_RodBiSegmentParameters
{
    real3 kBending;
    real kTwist;
    real2 kappaEq[Nstates];
    real tauEq[Nstates];
    real groundE[Nstates];
};

// theta0 and theta1 might be close to pi, leading to +- pi values
// this function compute the difference between the angles such as it is safely less that pi
__device__ inline rReal safeDiffTheta(rReal t0, rReal t1)
{
    auto dth = t1 - t0;
    if (dth >  M_PI) dth -= 2.0_rr * M_PI;
    if (dth < -M_PI) dth += 2.0_rr * M_PI;
    return dth;
}

__device__ inline rReal2 symmetricMatMult(const rReal3& A, const rReal2& x)
{
    return {A.x * x.x + A.y * x.y,
            A.y * x.x + A.z * x.y};
}

template <int Nstates>
__device__ inline rReal computeEnergy(rReal l, rReal2 kappa0, rReal2 kappa1, rReal tau, int state,
                                     const GPU_RodBiSegmentParameters<Nstates>& params)
{
    rReal2 dkappa0 = kappa0 - make_rReal2(params.kappaEq[state]);
    rReal2 dkappa1 = kappa1 - make_rReal2(params.kappaEq[state]);
    
    rReal2 Bkappa0 = symmetricMatMult(make_rReal3(params.kBending), dkappa0);
    rReal2 Bkappa1 = symmetricMatMult(make_rReal3(params.kBending), dkappa1);
    
    rReal Eb = 0.25_rr * l * (dot(dkappa0, Bkappa0) + dot(dkappa1, Bkappa1));
    
    rReal dtau = tau - params.tauEq[state];
    
    rReal Et = 0.5_rr * l * params.kTwist * dtau * dtau;

    return Eb + Et + params.groundE[state];
}


template <int Nstates>
struct BiSegment
{
    rReal3 e0, e1, t0, t1, dp0, dp1, bicur;
    rReal bicurFactor, e0inv, e1inv, linv, l;

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

        rReal le0 = length(e0);
        rReal le1 = length(e1);
        e0inv       = 1.0_rr / le0;
        e1inv       = 1.0_rr / le1;
        l           = 0.5_rr * (le0 + le1);
        linv        = 1.0_rr / l;
        bicurFactor = 1.0_rr / (le0 * le1 + dot(e0, e1));
        bicur       = (2.0_rr * bicurFactor) * cross(e0, e1);
    }

    __device__ inline rReal3 applyGrad0Bicur(const rReal3& v) const
    {
        return bicurFactor * (2.0_rr * cross(e0, v) + dot(e0, v) * bicur);
    }

    __device__ inline rReal3 applyGrad2Bicur(const rReal3& v) const
    {
        return bicurFactor * (2.0_rr * cross(e1, v) + dot(e1, v) * bicur);
    }

    __device__ inline void computeBendingForces(int state, const GPU_RodBiSegmentParameters<Nstates>& params,
                                                rReal3& fr0, rReal3& fr2, rReal3& fpm0, rReal3& fpm1) const
    {
        rReal dpt0 = dot(dp0, t0);
        rReal dpt1 = dot(dp1, t1);

        rReal3 t0_dp0 = cross(t0, dp0);
        rReal3 t1_dp1 = cross(t1, dp1);
    
        rReal3 dpPerp0 = dp0 - dpt0 * t0;
        rReal3 dpPerp1 = dp1 - dpt1 * t1;

        rReal dpPerp0inv = math::rsqrt(dot(dpPerp0, dpPerp0));
        rReal dpPerp1inv = math::rsqrt(dot(dpPerp1, dpPerp1));
    
        rReal2 kappa0 { +dpPerp0inv * linv * dot(bicur, t0_dp0),
                       -dpPerp0inv * linv * dot(bicur,    dp0)};

        rReal2 kappa1 { +dpPerp1inv * linv * dot(bicur, t1_dp1),
                       -dpPerp1inv * linv * dot(bicur,    dp1)};

        rReal2 dkappa0 = kappa0 - make_rReal2(params.kappaEq[state]);
        rReal2 dkappa1 = kappa1 - make_rReal2(params.kappaEq[state]);

        rReal2 Bkappa0 = symmetricMatMult(make_rReal3(params.kBending), dkappa0);
        rReal2 Bkappa1 = symmetricMatMult(make_rReal3(params.kBending), dkappa1);

        rReal Eb_linv = 0.25_rr * (dot(dkappa0, Bkappa0) + dot(dkappa1, Bkappa1));

        rReal3 grad0NormKappa0 = - 0.5_rr * linv * t0 - (e0inv * dpPerp0inv * dpPerp0inv * dpt0) * dpPerp0;
        rReal3 grad2NormKappa0 =   0.5_rr * linv * t1;

        rReal3 grad0NormKappa1 = - 0.5_rr * linv * t0;
        rReal3 grad2NormKappa1 =   0.5_rr * linv * t1 + (e1inv * dpPerp1inv * dpPerp1inv * dpt1) * dpPerp1;

        // 1. contributions of center line:    
        rReal3 baseGradKappa0x = cross(bicur, dp0) + dot(bicur, t0_dp0) * t0;
        rReal3 baseGradKappa1x = cross(bicur, dp1) + dot(bicur, t1_dp1) * t1;
    

        rReal3 grad0Kappa0x = kappa0.x * grad0NormKappa0 + dpPerp0inv * linv * (applyGrad0Bicur(t0_dp0) - e0inv * baseGradKappa0x);
        rReal3 grad2Kappa0x = kappa0.x * grad2NormKappa0 + dpPerp0inv * linv *  applyGrad2Bicur(t0_dp0);
        rReal3 grad0Kappa0y = kappa0.y * grad0NormKappa0 - dpPerp0inv * linv *  applyGrad0Bicur(   dp0);
        rReal3 grad2Kappa0y = kappa0.y * grad2NormKappa0 - dpPerp0inv * linv *  applyGrad2Bicur(   dp0);

        rReal3 grad0Kappa1x = kappa1.x * grad0NormKappa1 + dpPerp1inv * linv *  applyGrad0Bicur(t1_dp1);
        rReal3 grad2Kappa1x = kappa1.x * grad2NormKappa1 + dpPerp1inv * linv * (applyGrad2Bicur(t1_dp1) + e1inv * baseGradKappa1x);

        rReal3 grad0Kappa1y = kappa1.y * grad0NormKappa1 - dpPerp1inv * linv *  applyGrad0Bicur(   dp1);
        rReal3 grad2Kappa1y = kappa1.y * grad2NormKappa1 - dpPerp1inv * linv *  applyGrad2Bicur(   dp1);
    
        // 1.a contribution of kappa
        fr0 += 0.5_rr * l * (Bkappa0.x * grad0Kappa0x + Bkappa0.y * grad0Kappa0y  +  Bkappa1.x * grad0Kappa1x + Bkappa1.y * grad0Kappa1y);
        fr2 += 0.5_rr * l * (Bkappa0.x * grad2Kappa0x + Bkappa0.y * grad2Kappa0y  +  Bkappa1.x * grad2Kappa1x + Bkappa1.y * grad2Kappa1y);

        // 1.b contribution of l
        fr0 += ( 0.5_rr * Eb_linv) * t0;
        fr2 += (-0.5_rr * Eb_linv) * t1;

        // 2. contributions material frame:

        rReal3 baseGradKappaMF0 = (- dpPerp0inv * dpPerp0inv) * dpPerp0;
        rReal3 baseGradKappaMF1 = (- dpPerp1inv * dpPerp1inv) * dpPerp1;

        rReal3 gradKappaMF0x = kappa0.x * baseGradKappaMF0 + linv * dpPerp0inv * cross(bicur, t0);
        rReal3 gradKappaMF0y = kappa0.y * baseGradKappaMF0 - linv * dpPerp0inv * bicur;

        rReal3 gradKappaMF1x = kappa1.x * baseGradKappaMF1 + linv * dpPerp1inv * cross(bicur, t1);
        rReal3 gradKappaMF1y = kappa1.y * baseGradKappaMF1 - linv * dpPerp1inv * bicur;

        fpm0 += 0.5_rr * l * (Bkappa0.x * gradKappaMF0x + Bkappa0.y * gradKappaMF0y);
        fpm1 += 0.5_rr * l * (Bkappa1.x * gradKappaMF1x + Bkappa1.y * gradKappaMF1y);
    }

    __device__ inline void computeTwistForces(int state, const GPU_RodBiSegmentParameters<Nstates>& params,
                                              rReal3& fr0, rReal3& fr2, rReal3& fpm0, rReal3& fpm1) const
    {
        rReal4  Q = Quaternion::getFromVectorPair(t0, t1);
        rReal3 u0 = normalize(anyOrthogonal(t0));
        rReal3 u1 = normalize(Quaternion::rotate(u0, Q));
        
        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        rReal dpu0 = dot(dp0, u0);
        rReal dpv0 = dot(dp0, v0);

        rReal dpu1 = dot(dp1, u1);
        rReal dpv1 = dot(dp1, v1);

        rReal theta0 = math::atan2(dpv0, dpu0);
        rReal theta1 = math::atan2(dpv1, dpu1);
    
        rReal tau = safeDiffTheta(theta0, theta1) * linv;
        rReal dtau = tau - params.tauEq[state];

        // contribution from segment length on center line:
        
        rReal ftwistLFactor = 0.5_rr * params.kTwist * dtau * (tau + params.tauEq[state]);
        
        fr0 -= 0.5_rr * ftwistLFactor * t0;
        fr2 += 0.5_rr * ftwistLFactor * t1;

        // contribution from theta on center line:
        
        rReal dthetaFFactor = dtau * params.kTwist;
        
        fr0 += (0.5_rr * dthetaFFactor * e0inv) * bicur;
        fr2 -= (0.5_rr * dthetaFFactor * e1inv) * bicur;

        // contribution of theta on material frame:
        
        fpm0 += (dthetaFFactor / (dpu0*dpu0 + dpv0*dpv0)) * (dpv0 * u0 - dpu0 * v0);
        fpm1 += (dthetaFFactor / (dpu1*dpu1 + dpv1*dpv1)) * (dpu1 * v1 - dpv1 * u1);
    }

    

    __device__ inline void computeCurvatures(rReal2& kappa0, rReal2& kappa1) const
    {
        rReal dpt0 = dot(dp0, t0);
        rReal dpt1 = dot(dp1, t1);

        rReal3 t0_dp0 = cross(t0, dp0);
        rReal3 t1_dp1 = cross(t1, dp1);
    
        rReal3 dpPerp0 = dp0 - dpt0 * t0;
        rReal3 dpPerp1 = dp1 - dpt1 * t1;

        rReal dpPerp0inv = math::rsqrt(dot(dpPerp0, dpPerp0));
        rReal dpPerp1inv = math::rsqrt(dot(dpPerp1, dpPerp1));
    
        kappa0.x =   dpPerp0inv * linv * dot(bicur, t0_dp0);
        kappa0.y = - dpPerp0inv * linv * dot(bicur,    dp0);

        kappa1.x =   dpPerp1inv * linv * dot(bicur, t1_dp1);
        kappa1.y = - dpPerp1inv * linv * dot(bicur,    dp1);
    }

    __device__ inline void computeTorsion(rReal& tau) const
    {
        rReal4  Q = Quaternion::getFromVectorPair(t0, t1);
        rReal3 u0 = normalize(anyOrthogonal(t0));
        rReal3 u1 = normalize(Quaternion::rotate(u0, Q));

        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        rReal dpu0 = dot(dp0, u0);
        rReal dpv0 = dot(dp0, v0);

        rReal dpu1 = dot(dp1, u1);
        rReal dpv1 = dot(dp1, v1);

        rReal theta0 = math::atan2(dpv0, dpu0);
        rReal theta1 = math::atan2(dpv1, dpu1);
    
        tau = safeDiffTheta(theta0, theta1) * linv;
    }

    
    __device__ inline void computeCurvaturesGradients(rReal3& gradr0x, rReal3& gradr0y,
                                                      rReal3& gradr2x, rReal3& gradr2y,
                                                      rReal3& gradpm0x, rReal3& gradpm0y,
                                                      rReal3& gradpm1x, rReal3& gradpm1y) const
    {
        rReal dpt0 = dot(dp0, t0);
        rReal dpt1 = dot(dp1, t1);

        rReal3 t0_dp0 = cross(t0, dp0);
        rReal3 t1_dp1 = cross(t1, dp1);
    
        rReal3 dpPerp0 = dp0 - dpt0 * t0;
        rReal3 dpPerp1 = dp1 - dpt1 * t1;

        rReal dpPerp0inv = math::rsqrt(dot(dpPerp0, dpPerp0));
        rReal dpPerp1inv = math::rsqrt(dot(dpPerp1, dpPerp1));
    
        rReal2 kappa0 { +dpPerp0inv * linv * dot(bicur, t0_dp0),
                       -dpPerp0inv * linv * dot(bicur,    dp0)};

        rReal2 kappa1 { +dpPerp1inv * linv * dot(bicur, t1_dp1),
                       -dpPerp1inv * linv * dot(bicur,    dp1)};

        rReal3 grad0NormKappa0 = - 0.5_rr * linv * t0 - (e0inv * dpPerp0inv * dpPerp0inv * dpt0) * dpPerp0;
        rReal3 grad2NormKappa0 =   0.5_rr * linv * t1;

        rReal3 grad0NormKappa1 = - 0.5_rr * linv * t0;
        rReal3 grad2NormKappa1 =   0.5_rr * linv * t1 + (e1inv * dpPerp1inv * dpPerp1inv * dpt1) * dpPerp1;

        // 1. contributions of center line:    
        rReal3 baseGradKappa0x = cross(bicur, dp0) + dot(bicur, t0_dp0) * t0;
        rReal3 baseGradKappa1x = cross(bicur, dp1) + dot(bicur, t1_dp1) * t1;
    

        rReal3 grad0Kappa0x = kappa0.x * grad0NormKappa0 + dpPerp0inv * linv * (applyGrad0Bicur(t0_dp0) - e0inv * baseGradKappa0x);
        rReal3 grad2Kappa0x = kappa0.x * grad2NormKappa0 + dpPerp0inv * linv *  applyGrad2Bicur(t0_dp0);
        rReal3 grad0Kappa0y = kappa0.y * grad0NormKappa0 - dpPerp0inv * linv *  applyGrad0Bicur(   dp0);
        rReal3 grad2Kappa0y = kappa0.y * grad2NormKappa0 - dpPerp0inv * linv *  applyGrad2Bicur(   dp0);

        rReal3 grad0Kappa1x = kappa1.x * grad0NormKappa1 + dpPerp1inv * linv *  applyGrad0Bicur(t1_dp1);
        rReal3 grad2Kappa1x = kappa1.x * grad2NormKappa1 + dpPerp1inv * linv * (applyGrad2Bicur(t1_dp1) + e1inv * baseGradKappa1x);

        rReal3 grad0Kappa1y = kappa1.y * grad0NormKappa1 - dpPerp1inv * linv *  applyGrad0Bicur(   dp1);
        rReal3 grad2Kappa1y = kappa1.y * grad2NormKappa1 - dpPerp1inv * linv *  applyGrad2Bicur(   dp1);
    
        // 1.a contribution of kappa
        gradr0x = 0.5_rr * (grad0Kappa0x + grad0Kappa1x);
        gradr0y = 0.5_rr * (grad0Kappa0y + grad0Kappa1y);

        gradr2x = 0.5_rr * (grad2Kappa0x + grad2Kappa1x);
        gradr2y = 0.5_rr * (grad2Kappa0y + grad2Kappa1y);

        // 1.b contribution of l
        gradr0x += ( 0.5_rr * kappa0.x * linv) * t0;
        gradr0y += ( 0.5_rr * kappa0.y * linv) * t0;

        gradr2x += (-0.5_rr * kappa1.x * linv) * t1;
        gradr2y += (-0.5_rr * kappa1.y * linv) * t1;

        // 2. contributions material frame:

        rReal3 baseGradKappaMF0 = (- dpPerp0inv * dpPerp0inv) * dpPerp0;
        rReal3 baseGradKappaMF1 = (- dpPerp1inv * dpPerp1inv) * dpPerp1;

        gradpm0x = kappa0.x * baseGradKappaMF0 + linv * dpPerp0inv * cross(bicur, t0);
        gradpm0y = kappa0.y * baseGradKappaMF0 - linv * dpPerp0inv * bicur;

        gradpm1x = kappa1.x * baseGradKappaMF1 + linv * dpPerp1inv * cross(bicur, t1);
        gradpm1x = kappa1.y * baseGradKappaMF1 - linv * dpPerp1inv * bicur;
    }

    __device__ inline void computeTorsionGradients(rReal3& gradr0, rReal3& gradr2,
                                                   rReal3& gradpm0, rReal3& gradpm1) const
    {
        rReal4  Q = Quaternion::getFromVectorPair(t0, t1);
        rReal3 u0 = normalize(anyOrthogonal(t0));
        rReal3 u1 = normalize(Quaternion::rotate(u0, Q));
        
        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        rReal dpu0 = dot(dp0, u0);
        rReal dpv0 = dot(dp0, v0);

        rReal dpu1 = dot(dp1, u1);
        rReal dpv1 = dot(dp1, v1);

        rReal theta0 = math::atan2(dpv0, dpu0);
        rReal theta1 = math::atan2(dpv1, dpu1);
    
        rReal tau = safeDiffTheta(theta0, theta1) * linv;

        // contribution from segment length on center line:

        gradr0 =  0.5_rr * tau * linv * t0;
        gradr2 = -0.5_rr * tau * linv * t1;

        // contribution from theta on center line:
        
        gradr0 -= (linv * 0.5_rr * e0inv) * bicur;
        gradr2 += (linv * 0.5_rr * e1inv) * bicur;

        // contribution of theta on material frame:
        
        gradpm0 = (-linv / (dpu0*dpu0 + dpv0*dpv0)) * (dpv0 * u0 - dpu0 * v0);
        gradpm1 = (-linv / (dpu1*dpu1 + dpv1*dpv1)) * (dpu1 * v1 - dpv1 * u1);
    }

    __device__ inline rReal computeEnergy(int state, const GPU_RodBiSegmentParameters<Nstates>& params) const
    {
        rReal2 kappa0, kappa1;
        rReal tau;
        computeCurvatures(kappa0, kappa1);
        computeTorsion(tau);
        return ::computeEnergy(l, kappa0, kappa1, tau, state, params);
    }
};
