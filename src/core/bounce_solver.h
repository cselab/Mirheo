/*
 * bounce.h
 *
 *  Created on: May 8, 2017
 *      Author: alexeedm
 */

#pragma once

/**
 * Find alpha such that F( alpha ) = 0, 0 <= alpha <= 1
 */
template <typename Equation>
__device__ inline float2 solveLinSearch_verbose(Equation F, float a = 0.0f, float b = 1.0f, float tolerance = 1e-6f)
{
    // F is one dimensional equation
    // It returns value signed + or - depending on whether
    // coordinate is inside at the current time, or outside
    // Sign mapping to inside/outside is irrelevant

    const int maxNIters = 20;

    float va = F(a);
    float vb = F(b);

    float mid, vmid;

    // Check if the collision is there in the first place
    if (va*vb > 0.0f) return make_float2(-1.0f);

    int iters;
    for (iters=0; iters<maxNIters; iters++)
    {
        const float lambda = min( max(vb / (vb - va),  0.1f), 0.9f );  // va*l + (1-l)*vb = 0
        mid = a *lambda + b *(1.0f - lambda);
        vmid = F(mid);

        if (va * vmid < 0.0f)
        {
            vb = vmid;
            b  = mid;
        }
        else
        {
            va = vmid;
            a = mid;
        }

        if (fabsf(vmid) < tolerance)
            break;
    }

//    if (fabs(vmid) > tolerance)
//        printf("Equation not solved: %f --> %f, error: %f, best alpha: %f\n", F((float)0.0), F((float)1.0), vmid, mid);

    return {mid, vmid};
}

template <typename Equation>
__device__ inline float solveLinSearch(Equation F, float a = 0.0f, float b = 1.0f, float tolerance = 1e-6f)
{
    float2 res = solveLinSearch_verbose(F, a, b, tolerance);
    return res.x;
}



__device__ inline float2 solveQuadratic(float a, float b, float c)
{
    if (fabsf(a) == 0.0f)
    {
        if (fabsf(b) == 0.0f) return make_float2(1e20f);
        else return make_float2(-c/b, 1e20f);
    }

    const float D = b*b - 4.0f*a*c;
    if (D < 0.0f) return make_float2(1e20f);

    const float q = -0.5f * (b + copysignf(sqrtf(D), b));

    return make_float2(q/a, c/q);
}

template<typename F, typename F_prime>
__device__ inline float2 solveNewton(F f, F_prime f_prime, float x0, float tolerance = 1e-6f)
{
    const int maxNIters = 10;

    float x = x0;
    float val;
    for (int iter=0; iter<maxNIters; iter++)
    {
        val = f(x);
        if (fabsf(val) < tolerance) return {x, val};
        x = x - val / f_prime(x);
    }

    return {x, val};
}

template<typename F, typename F_prime>
__device__ inline float2 solveMbabane(F f, F_prime f_prime, float x0, float x1, float tolerance = 1e-6f)
{
    // Swaziland. epta.
    // https://arxiv.org/pdf/1210.5766.pdf
    const int maxNIters = 20;

    float y0, y1;
    y0 = f(x0);
    y1 = f(x1);

    for (int iter=0; iter<maxNIters; iter++)
    {
        float q = (y1 - y0) / (x1 - x0);
        float x = x0 - (x0 - x1) / (1.0f - y1/y0 * q / f_prime(x1));

        x0 = x1;
        x1 = x;

        y0 = y1;
        y1 = f(x);

        if (fabsf(y1) < tolerance) return {x1, y1};
    }

    return {x1, y1};
}


/** code copied from gsl library; gsl_poly_solve_cubic
https://www.gnu.org/software/gsl/
slightly adapted

solves x^3 + a*x^2 + b*x + c = 0
**/
//
//template<typename Real3, typename Real>
//__device__ inline Real3 solveCubic(Real a, Real b, Real c)
//{
//    auto q = (a * a - (Real)3.0 * b);
//    auto r = ((Real)2.0 * a * a * a - (Real)9.0 * a * b + (Real)27.0 * c);
//
//    auto Q = q * (Real)(1.0 / 9.0);
//    auto R = r * (Real)(1.0 / 54.0);
//
//    auto Q3 = Q * Q * Q;
//    auto R2 = R * R;
//
//    auto R2_Q3 = a*(b*c*(Real)(-1.0/6.0) + a*(b*b*(Real)(-1.0/108.0) + a*c*(Real)(1.0/27.0))) + (Real)0.25*c*c + (Real)(1.0/27.0) * b*b*b;
//
//    if (R == 0 && Q == 0)
//        return { -a * (Real)(1.0 / 3.0), (Real)1e+20, (Real)1e+20 };
//
//    if (R2_Q3 < 0)
//    {
//        Real sgnR = (R >= (Real)0.0 ? (Real)1.0 : (Real)-1.0);
//        auto ratio = sgnR * sqrt (R2 / Q3);
//        auto theta = acos (ratio);
//        auto norm = (Real)-2.0 * sqrt (Q);
//        auto a_3 = a * (Real)(1.0 / 3.0);
//
//        Real3 res;
//        res.x = norm * cos (theta * (Real)(1.0 / 3.0)) - a_3;
//        res.y = norm * cos ((theta + (Real)2.0 * (Real)M_PI) * (Real)(1.0 / 3.0)) - a_3;
//        res.z = norm * cos ((theta - (Real)2.0 * (Real)M_PI) * (Real)(1.0 / 3.0)) - a_3;
//
//        return res;
//    }
//    else
//    {
//        Real sgnR = (R >= 0.0 ? 1.0 : -1.0);
//        auto A = -sgnR * pow (fabs(R) + sqrt(R2_Q3), (Real)1.0/3.0);
//        auto B = Q / A ;
//        return { A + B - a * (Real)(1.0 / 3.0), (Real)1e20f, (Real)1e20f };
//    }
//}
//
//template<typename Real3, typename Real>
//__device__ inline Real3 solveCubic(Real a, Real b, Real c, Real d)
//{
//    if (fabs(a) < (Real)1e-15)
//    {
//        return make_float3(solveQuadratic(b, c, d), 1e20f);
//    }
//
//    return solveCubic<Real3>(b/a, c/a, d/a);
//}
//








