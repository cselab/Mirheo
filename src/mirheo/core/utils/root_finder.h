#pragma once

#include <cmath>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/// utilities to find the roots of 1D functions
namespace root_finder
{
/// basic structure that stores the information of a root
struct RootInfo
{
    real x;   ///< argument of the function
    real val; ///< value of the function at x
};

/// represent an interval
struct Bounds
{
    real lo; ///< lower bound
    real up; ///< upper bound
};

/// special value to represent an invalid root.
constexpr RootInfo invalidRoot {-666._r, -666._r};

/// comparison
__D__ inline bool operator==(RootInfo lhs, RootInfo rhs)
{
    return lhs.x == rhs.x && lhs.val == rhs.val;
}

/// divide a by b if |b| < eps, a/eps otherwise
__D__ static inline real safeDivide(real a, real b)
{
    constexpr real eps {1e-6_r};

    if (math::abs(b) < eps)
        return a / copysign(eps, b);

    return a / b;
}

/** \brief Find a root of a given function using bisection method
    \tparam Equation The equation type
    \param F the equation to solve
    \param limits the interval on which to solve the equation
    \param tolerance Stop the iterations when F(x) is less that this tolerance
    \return RootInfo object. return invalidRoot if it did not converge.
 */
template <typename Equation>
__D__ inline RootInfo linearSearchVerbose(Equation F, const Bounds& limits, real tolerance = 1e-6_r)
{
    constexpr int maxNIters = 20;

    real a {limits.lo};
    real b {limits.up};

    real va = F(a);
    real vb = F(b);

    real mid, vmid;

    // Check if the collision is there in the first place
    if (va*vb > 0.0_r)
        return invalidRoot;

    for (int iter = 0; iter < maxNIters; ++iter)
    {
        const real lambda = math::min( math::max(safeDivide(vb, vb - va),  0.1_r), 0.9_r );  // va*l + (1-l)*vb = 0
        mid = a * lambda + b * (1.0_r - lambda);
        vmid = F(mid);

        if (va * vmid < 0.0_r)
        {
            vb = vmid;
            b  = mid;
        }
        else
        {
            va = vmid;
            a = mid;
        }

        if (math::abs(vmid) < tolerance)
            break;
    }
    return {mid, vmid};
}

/// Same linearSearchVerbose(). Returns only the root.
template <typename Equation>
__D__ inline real linearSearch(Equation F, const Bounds& limits, real tolerance = 1e-6_r)
{
    const RootInfo ri = linearSearchVerbose(F, limits, tolerance);
    return ri.x;
}

/** \brief Find a root of a given function using Newton method
    \tparam F The function type
    \tparam FPrime The function derivative type
    \param f the function to find the root
    \param fPrime the derivative of \p f
    \param x0 the initial guess
    \param tolerance Stop the iterations when F(x) is less that this tolerance
    \return The obtained RootInfo object.
 */
template <typename F, typename FPrime>
__D__ inline RootInfo newton(F f, FPrime fPrime, real x0, real tolerance = 1e-6_r)
{
    constexpr int maxNIters = 10;

    real x {x0};
    real val;
    for (int iter = 0; iter < maxNIters; ++iter)
    {
        val = f(x);
        if (math::abs(val) < tolerance)
            return {x, val};
        x = x - safeDivide(val, fPrime(x));
    }

    return {x, val};
}
} // namespace root_finder

} // namespace mirheo
