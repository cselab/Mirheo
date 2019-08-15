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
    return {mid, vmid};
}

template <typename Equation>
__device__ inline float solveLinSearch(Equation F, float a = 0.0f, float b = 1.0f, float tolerance = 1e-6f)
{
    float2 res = solveLinSearch_verbose(F, a, b, tolerance);
    return res.x;
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

