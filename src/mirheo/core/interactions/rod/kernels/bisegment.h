#pragma once

#include "real.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/quaternion.h>
#include <mirheo/core/pvs/views/rv.h>

namespace mirheo
{

/** Parameters of the bisegment forces.
    \tparam Nstates Number of polymorphic states
 */
template<int Nstates>
struct GPU_RodBiSegmentParameters
{
    real3 kBending; ///< bending force coefficient
    real kTwist;    ///< torsion force coefficient
    real2 kappaEq[Nstates]; ///< equilibrium curvature along the material frame (one per state)
    real tauEq   [Nstates]; ///< equilibrium torsion (one per state)
    real groundE [Nstates]; ///< ground energy of each state
};

/** theta0 and theta1 might be close to pi, leading to +- pi values
    this function computes the difference between the angles such as it is safely less that pi
*/
__device__ inline rReal safeDiffTheta(rReal t0, rReal t1)
{
    auto dth = t1 - t0;
    if (dth >  M_PI) dth -= 2.0_rr * M_PI;
    if (dth < -M_PI) dth += 2.0_rr * M_PI;
    return dth;
}

/** Multiply a symmetric 2X2 matrix with a vector.
    \param [in] A The symmetrix 2X2 matrix
    \param [in] x The input vector
    \return A * x

    The matrix has is stored as A = (Axx, Axy, Azz).
 */
__device__ inline rReal2 symmetricMatMult(const rReal3& A, const rReal2& x)
{
    return {A.x * x.x + A.y * x.y,
            A.y * x.x + A.z * x.y};
}

/** Compute the elastic energy of a bisegment from its curvature and torsion
    \param [in] l The length of the bisegment
    \param [in] kappa0 The curvature along the first material direction
    \param [in] kappa1 The curvature along the second material direction
    \param [in] tau The torsion of the bisegment
    \param [in] state Polymorphic state
    \param [in] params Energy parameters
    \return The elastic energy
 */
template <int Nstates>
__device__ inline rReal computeEnergy(rReal l, rReal2 kappa0, rReal2 kappa1, rReal tau, int state,
                                     const GPU_RodBiSegmentParameters<Nstates>& params)
{
    const rReal2 dkappa0 = kappa0 - make_rReal2(params.kappaEq[state]);
    const rReal2 dkappa1 = kappa1 - make_rReal2(params.kappaEq[state]);

    const rReal2 Bkappa0 = symmetricMatMult(make_rReal3(params.kBending), dkappa0);
    const rReal2 Bkappa1 = symmetricMatMult(make_rReal3(params.kBending), dkappa1);

    const rReal Eb = 0.25_rr * l * (dot(dkappa0, Bkappa0) + dot(dkappa1, Bkappa1));

    const rReal dtau = tau - params.tauEq[state];

    const rReal Et = 0.5_rr * l * params.kTwist * dtau * dtau;

    return Eb + Et + params.groundE[state];
}

/** Helper class to compute elastic forces and energy on a bisegment
    \tparam Nstates Number of polymorphic states
 */
template <int Nstates>
struct BiSegment
{
    rReal3 e0;  ///< first segment
    rReal3 e1;  ///< second segment
    rReal3 t0;  ///< first segment direction
    rReal3 t1;  ///< second segment direction
    rReal3 dp0; ///< first material frame direction
    rReal3 dp1; ///< second material frame direction
    rReal3 bicur;  ///< bicurvature
    rReal bicurFactor;  ///< helper scalar to compute bicurvature
    rReal e0inv; ///< 1 / length of first segment
    rReal e1inv; ///< 1 / length of second segment
    rReal linv;  ///< 1 / l
    rReal l;     ///< average of the lengths of the two segments

    /** Fetch bisegment data and prepare helper quantities
     */
    __device__ inline BiSegment(const RVview& view, int start)
    {
        const auto r0  = fetchPosition(view, start + 0);
        const auto r1  = fetchPosition(view, start + 5);
        const auto r2  = fetchPosition(view, start + 10);

        const auto pm0 = fetchPosition(view, start + 1);
        const auto pp0 = fetchPosition(view, start + 2);
        const auto pm1 = fetchPosition(view, start + 6);
        const auto pp1 = fetchPosition(view, start + 7);

        e0 = r1 - r0;
        e1 = r2 - r1;

        t0 = normalize(e0);
        t1 = normalize(e1);

        dp0 = pp0 - pm0;
        dp1 = pp1 - pm1;

        const rReal le0 = length(e0);
        const rReal le1 = length(e1);
        e0inv       = 1.0_rr / le0;
        e1inv       = 1.0_rr / le1;
        l           = 0.5_rr * (le0 + le1);
        linv        = 1.0_rr / l;
        bicurFactor = 1.0_rr / (le0 * le1 + dot(e0, e1));
        bicur       = (2.0_rr * bicurFactor) * cross(e0, e1);
    }

    /// compute gradient of the bicurvature w.r.t. r0 times v
    __device__ inline rReal3 applyGrad0Bicur(const rReal3& v) const
    {
        return bicurFactor * (2.0_rr * cross(e0, v) + dot(e0, v) * bicur);
    }

    /// compute gradient of the bicurvature w.r.t. r0 times v
    __device__ inline rReal3 applyGrad2Bicur(const rReal3& v) const
    {
        return bicurFactor * (2.0_rr * cross(e1, v) + dot(e1, v) * bicur);
    }

    /** compute the bending forces acting on the bisegment particles
        \param [in] state Polymorphic state
        \param [in] params Elastic forces parameters
        \param [in,out] fr0 Force acting on r0
        \param [in,out] fr2 Force acting on r2
        \param [in,out] fpm0 Force acting on pm0
        \param [in,out] fpm1 Force acting on pm1

        This metho will add bending foces to the given variables.
        The other forces (on e.g. r1) can be computed by the symmetric nature of the model.
     */
    __device__ inline void computeBendingForces(int state, const GPU_RodBiSegmentParameters<Nstates>& params,
                                                rReal3& fr0, rReal3& fr2, rReal3& fpm0, rReal3& fpm1) const
    {
        const rReal dpt0 = dot(dp0, t0);
        const rReal dpt1 = dot(dp1, t1);

        const rReal3 t0_dp0 = cross(t0, dp0);
        const rReal3 t1_dp1 = cross(t1, dp1);

        const rReal3 dpPerp0 = dp0 - dpt0 * t0;
        const rReal3 dpPerp1 = dp1 - dpt1 * t1;

        const rReal dpPerp0inv = math::rsqrt(dot(dpPerp0, dpPerp0));
        const rReal dpPerp1inv = math::rsqrt(dot(dpPerp1, dpPerp1));

        const rReal2 kappa0 { +dpPerp0inv * linv * dot(bicur, t0_dp0),
                              -dpPerp0inv * linv * dot(bicur,    dp0)};

        const rReal2 kappa1 { +dpPerp1inv * linv * dot(bicur, t1_dp1),
                              -dpPerp1inv * linv * dot(bicur,    dp1)};

        const rReal2 dkappa0 = kappa0 - make_rReal2(params.kappaEq[state]);
        const rReal2 dkappa1 = kappa1 - make_rReal2(params.kappaEq[state]);

        const rReal2 Bkappa0 = symmetricMatMult(make_rReal3(params.kBending), dkappa0);
        const rReal2 Bkappa1 = symmetricMatMult(make_rReal3(params.kBending), dkappa1);

        const rReal Eb_linv = 0.25_rr * (dot(dkappa0, Bkappa0) + dot(dkappa1, Bkappa1));

        const rReal3 grad0NormKappa0 = - 0.5_rr * linv * t0 - (e0inv * dpPerp0inv * dpPerp0inv * dpt0) * dpPerp0;
        const rReal3 grad2NormKappa0 =   0.5_rr * linv * t1;

        const rReal3 grad0NormKappa1 = - 0.5_rr * linv * t0;
        const rReal3 grad2NormKappa1 =   0.5_rr * linv * t1 + (e1inv * dpPerp1inv * dpPerp1inv * dpt1) * dpPerp1;

        // 1. contributions of center line:
        const rReal3 baseGradKappa0x = cross(bicur, dp0) + dot(bicur, t0_dp0) * t0;
        const rReal3 baseGradKappa1x = cross(bicur, dp1) + dot(bicur, t1_dp1) * t1;


        const rReal3 grad0Kappa0x = kappa0.x * grad0NormKappa0 + dpPerp0inv * linv * (applyGrad0Bicur(t0_dp0) - e0inv * baseGradKappa0x);
        const rReal3 grad2Kappa0x = kappa0.x * grad2NormKappa0 + dpPerp0inv * linv *  applyGrad2Bicur(t0_dp0);
        const rReal3 grad0Kappa0y = kappa0.y * grad0NormKappa0 - dpPerp0inv * linv *  applyGrad0Bicur(   dp0);
        const rReal3 grad2Kappa0y = kappa0.y * grad2NormKappa0 - dpPerp0inv * linv *  applyGrad2Bicur(   dp0);

        const rReal3 grad0Kappa1x = kappa1.x * grad0NormKappa1 + dpPerp1inv * linv *  applyGrad0Bicur(t1_dp1);
        const rReal3 grad2Kappa1x = kappa1.x * grad2NormKappa1 + dpPerp1inv * linv * (applyGrad2Bicur(t1_dp1) + e1inv * baseGradKappa1x);

        const rReal3 grad0Kappa1y = kappa1.y * grad0NormKappa1 - dpPerp1inv * linv *  applyGrad0Bicur(   dp1);
        const rReal3 grad2Kappa1y = kappa1.y * grad2NormKappa1 - dpPerp1inv * linv *  applyGrad2Bicur(   dp1);

        // 1.a contribution of kappa
        fr0 += 0.5_rr * l * (Bkappa0.x * grad0Kappa0x + Bkappa0.y * grad0Kappa0y  +  Bkappa1.x * grad0Kappa1x + Bkappa1.y * grad0Kappa1y);
        fr2 += 0.5_rr * l * (Bkappa0.x * grad2Kappa0x + Bkappa0.y * grad2Kappa0y  +  Bkappa1.x * grad2Kappa1x + Bkappa1.y * grad2Kappa1y);

        // 1.b contribution of l
        fr0 += ( 0.5_rr * Eb_linv) * t0;
        fr2 += (-0.5_rr * Eb_linv) * t1;

        // 2. contributions material frame:

        const rReal3 baseGradKappaMF0 = (- dpPerp0inv * dpPerp0inv) * dpPerp0;
        const rReal3 baseGradKappaMF1 = (- dpPerp1inv * dpPerp1inv) * dpPerp1;

        const rReal3 gradKappaMF0x = kappa0.x * baseGradKappaMF0 + linv * dpPerp0inv * cross(bicur, t0);
        const rReal3 gradKappaMF0y = kappa0.y * baseGradKappaMF0 - linv * dpPerp0inv * bicur;

        const rReal3 gradKappaMF1x = kappa1.x * baseGradKappaMF1 + linv * dpPerp1inv * cross(bicur, t1);
        const rReal3 gradKappaMF1y = kappa1.y * baseGradKappaMF1 - linv * dpPerp1inv * bicur;

        fpm0 += 0.5_rr * l * (Bkappa0.x * gradKappaMF0x + Bkappa0.y * gradKappaMF0y);
        fpm1 += 0.5_rr * l * (Bkappa1.x * gradKappaMF1x + Bkappa1.y * gradKappaMF1y);
    }

    /** compute the torsion forces acting on the bisegment particles
        \param [in] state Polymorphic state
        \param [in] params Elastic forces parameters
        \param [in,out] fr0 Force acting on r0
        \param [in,out] fr2 Force acting on r2
        \param [in,out] fpm0 Force acting on pm0
        \param [in,out] fpm1 Force acting on pm1

        This metho will add twist foces to the given variables.
        The other forces (on e.g. r1) can be computed by the symmetric nature of the model.
    */
        __device__ inline void computeTwistForces(int state, const GPU_RodBiSegmentParameters<Nstates>& params,
                                              rReal3& fr0, rReal3& fr2, rReal3& fpm0, rReal3& fpm1) const
    {
        const auto Q = Quaternion<rReal>::createFromVectors(t0, t1);
        const rReal3 u0 = normalize(anyOrthogonal(t0));
        const rReal3 u1 = normalize(Q.rotate(u0));

        const auto v0 = cross(t0, u0);
        const auto v1 = cross(t1, u1);

        const rReal dpu0 = dot(dp0, u0);
        const rReal dpv0 = dot(dp0, v0);

        const rReal dpu1 = dot(dp1, u1);
        const rReal dpv1 = dot(dp1, v1);

        const rReal theta0 = math::atan2(dpv0, dpu0);
        const rReal theta1 = math::atan2(dpv1, dpu1);

        const rReal tau = safeDiffTheta(theta0, theta1) * linv;
        const rReal dtau = tau - params.tauEq[state];

        // contribution from segment length on center line:

        const rReal ftwistLFactor = 0.5_rr * params.kTwist * dtau * (tau + params.tauEq[state]);

        fr0 -= 0.5_rr * ftwistLFactor * t0;
        fr2 += 0.5_rr * ftwistLFactor * t1;

        // contribution from theta on center line:

        const rReal dthetaFFactor = dtau * params.kTwist;

        fr0 += (0.5_rr * dthetaFFactor * e0inv) * bicur;
        fr2 -= (0.5_rr * dthetaFFactor * e1inv) * bicur;

        // contribution of theta on material frame:

        fpm0 += (dthetaFFactor / (dpu0*dpu0 + dpv0*dpv0)) * (dpv0 * u0 - dpu0 * v0);
        fpm1 += (dthetaFFactor / (dpu1*dpu1 + dpv1*dpv1)) * (dpu1 * v1 - dpv1 * u1);
    }


    /** Compute the curvatures along the material frames on each segment.
        \param [out] kappa0 Curvature on the first segment
        \param [out] kappa1 Curvature on the second segment
     */
    __device__ inline void computeCurvatures(rReal2& kappa0, rReal2& kappa1) const
    {
        const rReal dpt0 = dot(dp0, t0);
        const rReal dpt1 = dot(dp1, t1);

        const rReal3 t0_dp0 = cross(t0, dp0);
        const rReal3 t1_dp1 = cross(t1, dp1);

        const rReal3 dpPerp0 = dp0 - dpt0 * t0;
        const rReal3 dpPerp1 = dp1 - dpt1 * t1;

        const rReal dpPerp0inv = math::rsqrt(dot(dpPerp0, dpPerp0));
        const rReal dpPerp1inv = math::rsqrt(dot(dpPerp1, dpPerp1));

        kappa0.x =   dpPerp0inv * linv * dot(bicur, t0_dp0);
        kappa0.y = - dpPerp0inv * linv * dot(bicur,    dp0);

        kappa1.x =   dpPerp1inv * linv * dot(bicur, t1_dp1);
        kappa1.y = - dpPerp1inv * linv * dot(bicur,    dp1);
    }

    /** Compute the torsion along the bisegment
        \param [out] tau Torsion
     */
    __device__ inline void computeTorsion(rReal& tau) const
    {
        const auto Q = Quaternion<rReal>::createFromVectors(t0, t1);
        const rReal3 u0 = normalize(anyOrthogonal(t0));
        const rReal3 u1 = normalize(Q.rotate(u0));

        const auto v0 = cross(t0, u0);
        const auto v1 = cross(t1, u1);

        const rReal dpu0 = dot(dp0, u0);
        const rReal dpv0 = dot(dp0, v0);

        const rReal dpu1 = dot(dp1, u1);
        const rReal dpv1 = dot(dp1, v1);

        const rReal theta0 = math::atan2(dpv0, dpu0);
        const rReal theta1 = math::atan2(dpv1, dpu1);

        tau = safeDiffTheta(theta0, theta1) * linv;
    }

    /// compute gradients of curvature term w.r.t. particle positions (see drivers)
    __device__ inline void computeCurvaturesGradients(rReal3& gradr0x, rReal3& gradr0y,
                                                      rReal3& gradr2x, rReal3& gradr2y,
                                                      rReal3& gradpm0x, rReal3& gradpm0y,
                                                      rReal3& gradpm1x, rReal3& gradpm1y) const
    {
        const rReal dpt0 = dot(dp0, t0);
        const rReal dpt1 = dot(dp1, t1);

        const rReal3 t0_dp0 = cross(t0, dp0);
        const rReal3 t1_dp1 = cross(t1, dp1);

        const rReal3 dpPerp0 = dp0 - dpt0 * t0;
        const rReal3 dpPerp1 = dp1 - dpt1 * t1;

        const rReal dpPerp0inv = math::rsqrt(dot(dpPerp0, dpPerp0));
        const rReal dpPerp1inv = math::rsqrt(dot(dpPerp1, dpPerp1));

        const rReal2 kappa0 { +dpPerp0inv * linv * dot(bicur, t0_dp0),
                              -dpPerp0inv * linv * dot(bicur,    dp0)};

        const rReal2 kappa1 { +dpPerp1inv * linv * dot(bicur, t1_dp1),
                              -dpPerp1inv * linv * dot(bicur,    dp1)};

        const rReal3 grad0NormKappa0 = - 0.5_rr * linv * t0 - (e0inv * dpPerp0inv * dpPerp0inv * dpt0) * dpPerp0;
        const rReal3 grad2NormKappa0 =   0.5_rr * linv * t1;

        const rReal3 grad0NormKappa1 = - 0.5_rr * linv * t0;
        const rReal3 grad2NormKappa1 =   0.5_rr * linv * t1 + (e1inv * dpPerp1inv * dpPerp1inv * dpt1) * dpPerp1;

        // 1. contributions of center line:
        const rReal3 baseGradKappa0x = cross(bicur, dp0) + dot(bicur, t0_dp0) * t0;
        const rReal3 baseGradKappa1x = cross(bicur, dp1) + dot(bicur, t1_dp1) * t1;


        const rReal3 grad0Kappa0x = kappa0.x * grad0NormKappa0 + dpPerp0inv * linv * (applyGrad0Bicur(t0_dp0) - e0inv * baseGradKappa0x);
        const rReal3 grad2Kappa0x = kappa0.x * grad2NormKappa0 + dpPerp0inv * linv *  applyGrad2Bicur(t0_dp0);
        const rReal3 grad0Kappa0y = kappa0.y * grad0NormKappa0 - dpPerp0inv * linv *  applyGrad0Bicur(   dp0);
        const rReal3 grad2Kappa0y = kappa0.y * grad2NormKappa0 - dpPerp0inv * linv *  applyGrad2Bicur(   dp0);

        const rReal3 grad0Kappa1x = kappa1.x * grad0NormKappa1 + dpPerp1inv * linv *  applyGrad0Bicur(t1_dp1);
        const rReal3 grad2Kappa1x = kappa1.x * grad2NormKappa1 + dpPerp1inv * linv * (applyGrad2Bicur(t1_dp1) + e1inv * baseGradKappa1x);

        const rReal3 grad0Kappa1y = kappa1.y * grad0NormKappa1 - dpPerp1inv * linv *  applyGrad0Bicur(   dp1);
        const rReal3 grad2Kappa1y = kappa1.y * grad2NormKappa1 - dpPerp1inv * linv *  applyGrad2Bicur(   dp1);

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

        const rReal3 baseGradKappaMF0 = (- dpPerp0inv * dpPerp0inv) * dpPerp0;
        const rReal3 baseGradKappaMF1 = (- dpPerp1inv * dpPerp1inv) * dpPerp1;

        gradpm0x = kappa0.x * baseGradKappaMF0 + linv * dpPerp0inv * cross(bicur, t0);
        gradpm0y = kappa0.y * baseGradKappaMF0 - linv * dpPerp0inv * bicur;

        gradpm1x = kappa1.x * baseGradKappaMF1 + linv * dpPerp1inv * cross(bicur, t1);
        gradpm1x = kappa1.y * baseGradKappaMF1 - linv * dpPerp1inv * bicur;
    }

    /// compute gradients of torsion term w.r.t. particle positions (see drivers)
    __device__ inline void computeTorsionGradients(rReal3& gradr0, rReal3& gradr2,
                                                   rReal3& gradpm0, rReal3& gradpm1) const
    {
        const auto Q = Quaternion<rReal>::createFromVectors(t0, t1);
        const rReal3 u0 = normalize(anyOrthogonal(t0));
        const rReal3 u1 = normalize(Q.rotate(u0));

        const auto v0 = cross(t0, u0);
        const auto v1 = cross(t1, u1);

        const rReal dpu0 = dot(dp0, u0);
        const rReal dpv0 = dot(dp0, v0);

        const rReal dpu1 = dot(dp1, u1);
        const rReal dpv1 = dot(dp1, v1);

        const rReal theta0 = math::atan2(dpv0, dpu0);
        const rReal theta1 = math::atan2(dpv1, dpu1);

        const rReal tau = safeDiffTheta(theta0, theta1) * linv;

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

    /// Compute the energy of the bisegment
    __device__ inline rReal computeEnergy(int state, const GPU_RodBiSegmentParameters<Nstates>& params) const
    {
        rReal2 kappa0, kappa1;
        rReal tau;
        computeCurvatures(kappa0, kappa1);
        computeTorsion(tau);
        return mirheo::computeEnergy(l, kappa0, kappa1, tau, state, params);
    }
};

} // namespace mirheo
