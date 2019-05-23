#pragma once

#include "real.h"

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/pvs/views/rv.h>

template<int Nstates>
struct GPU_RodBiSegmentParameters
{
    float3 kBending;
    float kTwist;
    float2 omegaEq[Nstates];
    float tauEq[Nstates];
    float groundE[Nstates];
};

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

template <int Nstates>
struct BiSegment
{
    real3 e0, e1, t0, t1, dp0, dp1, bicur;
    real bicurFactor, e0inv, e1inv, linv, l;
    int state;

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
        e0inv       = 1.0_r / le0;
        e1inv       = 1.0_r / le1;
        l           = 0.5_r * (le0 + le1);
        linv        = 1.0_r / l;
        bicurFactor = 1.0_r / (le0 * le1 + dot(e0, e1));
        bicur       = (2.0_r * bicurFactor) * cross(e0, e1);
    }

    __device__ inline real3 applyGrad0Bicur(const real3& v) const
    {
        return bicurFactor * (2.0_r * cross(e0, v) + dot(e0, v) * bicur);
    }

    __device__ inline real3 applyGrad2Bicur(const real3& v) const
    {
        return bicurFactor * (2.0_r * cross(e1, v) + dot(e1, v) * bicur);
    }

    __device__ inline void computeBendingForces(int state, const GPU_RodBiSegmentParameters<Nstates>& params,
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
    
        real2 omega0 { +dpPerp0inv * linv * dot(bicur, t0_dp0),
                       -dpPerp0inv * linv * dot(bicur,    dp0)};

        real2 omega1 { +dpPerp1inv * linv * dot(bicur, t1_dp1),
                       -dpPerp1inv * linv * dot(bicur,    dp1)};

        real2 domega0 = omega0 - make_real2(params.omegaEq[state]);
        real2 domega1 = omega1 - make_real2(params.omegaEq[state]);

        real2 Bomega0 = symmetricMatMult(make_real3(params.kBending), domega0);
        real2 Bomega1 = symmetricMatMult(make_real3(params.kBending), domega1);

        real Eb_linv = 0.25_r * (dot(domega0, Bomega0) + dot(domega1, Bomega1));

        real3 grad0NormOmega0 = - 0.5_r * linv * t0 - (e0inv * dpPerp0inv * dpPerp0inv * dpt0) * dpPerp0;
        real3 grad2NormOmega0 =   0.5_r * linv * t1;

        real3 grad0NormOmega1 = - 0.5_r * linv * t0;
        real3 grad2NormOmega1 =   0.5_r * linv * t1 + (e1inv * dpPerp1inv * dpPerp1inv * dpt1) * dpPerp1;

        // 1. contributions of center line:    
        real3 baseGradOmega0x = cross(bicur, dp0) + dot(bicur, t0_dp0) * t0;
        real3 baseGradOmega1x = cross(bicur, dp1) + dot(bicur, t1_dp1) * t1;
    

        real3 grad0Omega0x = omega0.x * grad0NormOmega0 + dpPerp0inv * linv * (applyGrad0Bicur(t0_dp0) - e0inv * baseGradOmega0x);
        real3 grad2Omega0x = omega0.x * grad2NormOmega0 + dpPerp0inv * linv *  applyGrad2Bicur(t0_dp0);
        real3 grad0Omega0y = omega0.y * grad0NormOmega0 - dpPerp0inv * linv *  applyGrad0Bicur(   dp0);
        real3 grad2Omega0y = omega0.y * grad2NormOmega0 - dpPerp0inv * linv *  applyGrad2Bicur(   dp0);

        real3 grad0Omega1x = omega1.x * grad0NormOmega1 + dpPerp1inv * linv *  applyGrad0Bicur(t1_dp1);
        real3 grad2Omega1x = omega1.x * grad2NormOmega1 + dpPerp1inv * linv * (applyGrad2Bicur(t1_dp1) + e1inv * baseGradOmega1x);

        real3 grad0Omega1y = omega1.y * grad0NormOmega1 - dpPerp1inv * linv *  applyGrad0Bicur(   dp1);
        real3 grad2Omega1y = omega1.y * grad2NormOmega1 - dpPerp1inv * linv *  applyGrad2Bicur(   dp1);
    
        // 1.a contribution of omega
        fr0 += 0.5_r * l * (Bomega0.x * grad0Omega0x + Bomega0.y * grad0Omega0y  +  Bomega1.x * grad0Omega1x + Bomega1.y * grad0Omega1y);
        fr2 += 0.5_r * l * (Bomega0.x * grad2Omega0x + Bomega0.y * grad2Omega0y  +  Bomega1.x * grad2Omega1x + Bomega1.y * grad2Omega1y);

        // 1.b contribution of l
        fr0 += ( 0.5_r * Eb_linv) * t0;
        fr2 += (-0.5_r * Eb_linv) * t1;

        // 2. contributions material frame:

        real3 baseGradOmegaMF0 = (- dpPerp0inv * dpPerp0inv) * dpPerp0;
        real3 baseGradOmegaMF1 = (- dpPerp1inv * dpPerp1inv) * dpPerp1;

        real3 gradOmegaMF0x = omega0.x * baseGradOmegaMF0 + linv * dpPerp0inv * cross(bicur, t0);
        real3 gradOmegaMF0y = omega0.y * baseGradOmegaMF0 - linv * dpPerp0inv * bicur;

        real3 gradOmegaMF1x = omega1.x * baseGradOmegaMF1 + linv * dpPerp1inv * cross(bicur, t1);
        real3 gradOmegaMF1y = omega1.y * baseGradOmegaMF1 - linv * dpPerp1inv * bicur;

        fpm0 += 0.5_r * l * (Bomega0.x * gradOmegaMF0x + Bomega0.y * gradOmegaMF0y);
        fpm1 += 0.5_r * l * (Bomega1.x * gradOmegaMF1x + Bomega1.y * gradOmegaMF1y);
    }

    __device__ inline void computeTwistForces(int state, const GPU_RodBiSegmentParameters<Nstates>& params,
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
    
        real tau = safeDiffTheta(theta0, theta1) * linv;
        real dtau = tau - params.tauEq[state];

        real ftwistLFactor = 0.5_r * params.kTwist * dtau * (tau + params.tauEq[state]);

        fr0 -= 0.5_r * ftwistLFactor * t0;
        fr2 += 0.5_r * ftwistLFactor * t1;

        real dthetaFFactor = dtau * params.kTwist;

        fr0 += (0.5_r * dthetaFFactor * e0inv) * bicur;
        fr2 -= (0.5_r * dthetaFFactor * e1inv) * bicur;

        fpm0 += (dthetaFFactor / (dpu0*dpu0 + dpv0*dpv0)) * (dpv0 * u0 - dpu0 * v0);
        fpm1 += (dthetaFFactor / (dpu1*dpu1 + dpv1*dpv1)) * (dpu1 * v1 - dpv1 * u1);
    }

    __device__ inline real computeEnergy(int state, const GPU_RodBiSegmentParameters<Nstates>& params,
                                         const real3& u0, const real3& u1) const
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

        real2 domega0 = omega0 - make_real2(params.omegaEq[state]);
        real2 domega1 = omega1 - make_real2(params.omegaEq[state]);

        real2 Bomega0 = symmetricMatMult(make_real3(params.kBending), domega0);
        real2 Bomega1 = symmetricMatMult(make_real3(params.kBending), domega1);

        real Eb = 0.5_r * linv * (dot(domega0, Bomega0) + dot(domega1, Bomega1));


        auto v0 = cross(t0, u0);
        auto v1 = cross(t1, u1);

        real dpu0 = dot(dp0, u0);
        real dpv0 = dot(dp0, v0);

        real dpu1 = dot(dp1, u1);
        real dpv1 = dot(dp1, v1);

        real theta0 = atan2(dpv0, dpu0);
        real theta1 = atan2(dpv1, dpu1);
    
        real tau = safeDiffTheta(theta0, theta1) * linv;
        real dtau = tau - params.tauEq[state];

        real Et = params.kTwist / linv * dtau * dtau;

        return Eb + Et + params.groundE[state];
    }
};
