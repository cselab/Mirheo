#pragma once

#include <core/celllist.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/helper_math.h>
#include <core/utils/root_finder.h>

namespace BounceKernels
{

template <typename InsideWallChecker>
__device__ inline real3 rescue(real3 candidate, real dt, real tol, int seed, const InsideWallChecker& checker)
{
    const int maxIters = 100;
    const real factor = 5.0_r * dt;

    for (int i = 0; i < maxIters; i++)
    {
        const real v = checker(candidate);
        if (v < -tol) break;

        real3 rndShift;
        rndShift.x = Saru::mean0var1(candidate.x - math::floor(candidate.x), seed+i, seed*seed);
        rndShift.y = Saru::mean0var1(rndShift.x,                             seed+i, seed*seed);
        rndShift.z = Saru::mean0var1(rndShift.y,                             seed+i, seed*seed);

        if (checker(candidate + factor * rndShift) < v)
            candidate += factor * rndShift;
    }

    return candidate;
}

template <typename InsideWallChecker, typename VelocityField>
__global__ void sdfBounce(PVviewWithOldParticles view, CellListInfo cinfo,
                          const int *wallCells, const int nWallCells, const real dt,
                          const InsideWallChecker checker,
                          const VelocityField velField,
                          double3 *totalForce)
{
    const real insideTolerance = 2e-6_r;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    real3 localForce{0._r, 0._r, 0._r};
    
    if (tid < nWallCells)
    {
        const int cid = wallCells[tid];
        const int pstart = cinfo.cellStarts[cid];
        const int pend   = cinfo.cellStarts[cid+1];

        for (int pid = pstart; pid < pend; pid++)
        {
            Particle p(view.readParticle(pid));
            if (checker(p.r) <= -insideTolerance) continue;

            const auto rOld = view.readOldPosition(pid);
            const real3 dr = p.r - rOld;

            constexpr RootFinder::Bounds limits {0._r, 1._r};
            const real alpha = RootFinder::linearSearch([=] (real lambda)
            {
                return checker(rOld + dr*lambda) + insideTolerance;
            }, limits);

            real3 candidate = (alpha >= limits.lo) ? rOld + alpha * dr : rOld;
            candidate = rescue(candidate, dt, insideTolerance, p.i1, checker);

            const real3 uWall = velField(p.r);
            const real3 unew = 2.0_r * uWall - p.u;

            localForce += (p.u - unew) * (view.mass / dt); // force exerted by the particle on the wall

            p.r = candidate;
            p.u = unew;
                           
            view.writeParticle(pid, p);
        }        
    }

    localForce = warpReduce(localForce, [](real a, real b){return a+b;});
    
    if ((laneId() == 0) && (length(localForce) > 1e-8_r))
        atomicAdd(totalForce, make_double3(localForce));

}

} // namespace BounceKernels
