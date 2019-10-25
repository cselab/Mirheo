#pragma once

#include <cmath>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

namespace RootFinder
{
struct RootInfo
{
    real x;
    real val;
};

struct Bounds
{
    real lo, up;
};

constexpr RootInfo invalidRoot {-666.f, -666.f};

__D__ inline bool operator==(RootInfo lhs, RootInfo rhs)
{
    return lhs.x == rhs.x && lhs.val == rhs.val;
}

/**
 * Find alpha such that F( alpha ) = 0, 0 <= alpha <= 1
 */
template <typename Equation>
__D__ inline RootInfo linearSearchVerbose(Equation F, const Bounds& limits, real tolerance = 1e-6f)
{
    // F is one dimensional equation
    // It returns value signed + or - depending on whether
    // coordinate is inside at the current time, or outside
    // Sign mapping to inside/outside is irrelevant

    constexpr int maxNIters = 20;

    real a {limits.lo};
    real b {limits.up};

    real va = F(a);
    real vb = F(b);

    real mid, vmid;

    // Check if the collision is there in the first place
    if (va*vb > 0.0f)
        return invalidRoot;

    for (int iter = 0; iter < maxNIters; ++iter)
    {
        const real lambda = math::min( math::max(vb / (vb - va),  0.1f), 0.9f );  // va*l + (1-l)*vb = 0
        mid = a * lambda + b * (1.0f - lambda);
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

        if (math::abs(vmid) < tolerance)
            break;
    }
    return {mid, vmid};
}

template <typename Equation>
__D__ inline real linearSearch(Equation F, const Bounds& limits, real tolerance = 1e-6f)
{
    const RootInfo ri = linearSearchVerbose(F, limits, tolerance);
    return ri.x;
}

template <typename F, typename F_prime>
__D__ inline RootInfo newton(F f, F_prime f_prime, real x0, real tolerance = 1e-6f)
{
    constexpr int maxNIters = 10;

    real x {x0};
    real val;
    for (int iter = 0; iter < maxNIters; ++iter)
    {
        val = f(x);
        if (math::abs(val) < tolerance)
            return {x, val};
        x = x - val / f_prime(x);
    }

    return {x, val};
}
} // namespace RootFinder
