#pragma once

#include <cmath>
#include <core/utils/cpu_gpu_defines.h>

namespace RootFinder
{
struct RootInfo
{
    float x;
    float val;
};

struct Bounds
{
    float lo, up;
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
__D__ inline RootInfo linearSearchVerbose(Equation F, const Bounds& limits, float tolerance = 1e-6f)
{
    // F is one dimensional equation
    // It returns value signed + or - depending on whether
    // coordinate is inside at the current time, or outside
    // Sign mapping to inside/outside is irrelevant

    constexpr int maxNIters = 20;

    float a {limits.lo};
    float b {limits.up};

    float va = F(a);
    float vb = F(b);

    float mid, vmid;

    // Check if the collision is there in the first place
    if (va*vb > 0.0f)
        return invalidRoot;

    for (int iter = 0; iter < maxNIters; ++iter)
    {
        const float lambda = fminf( fmaxf(vb / (vb - va),  0.1f), 0.9f );  // va*l + (1-l)*vb = 0
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
__D__ inline float linearSearch(Equation F, const Bounds& limits, float tolerance = 1e-6f)
{
    const RootInfo ri = linearSearchVerbose(F, limits, tolerance);
    return ri.x;
}

template <typename F, typename F_prime>
__D__ inline RootInfo newton(F f, F_prime f_prime, float x0, float tolerance = 1e-6f)
{
    constexpr int maxNIters = 10;

    float x {x0};
    float val;
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
