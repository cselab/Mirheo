#pragma once

#include <core/celllist.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>

namespace bounceKernels
{

template <typename InsideWallChecker>
__device__ inline float3 rescue(float3 candidate, float dt, float tol, int id, const InsideWallChecker& checker)
{
    const int maxIters = 100;
    const float factor = 5.0f * dt;

    for (int i = 0; i < maxIters; i++)
    {
        float v = checker(candidate);
        if (v < -tol) break;

        float3 rndShift;
        rndShift.x = Saru::mean0var1(candidate.x - floorf(candidate.x), id+i, id*id);
        rndShift.y = Saru::mean0var1(rndShift.x,                        id+i, id*id);
        rndShift.z = Saru::mean0var1(rndShift.y,                        id+i, id*id);

        if (checker(candidate + factor * rndShift) < v)
            candidate += factor * rndShift;
    }

    return candidate;
}

template <typename InsideWallChecker, typename VelocityField>
__global__ void sdfBounce(PVviewWithOldParticles view, CellListInfo cinfo,
                          const int *wallCells, const int nWallCells, const float dt,
                          const InsideWallChecker checker,
                          const VelocityField velField,
                          double3 *totalForce)
{
    const float insideTolerance = 2e-6f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float3 localForce{0.f, 0.f, 0.f};
    
    if (tid < nWallCells)
    {
        const int cid = wallCells[tid];
        const int pstart = cinfo.cellStarts[cid];
        const int pend   = cinfo.cellStarts[cid+1];

        for (int pid = pstart; pid < pend; pid++)
        {
            Particle p(view.particles, pid);
            if (checker(p.r) <= -insideTolerance) continue;

            Particle pOld(view.old_particles, pid);
            float3 dr = p.r - pOld.r;

            const float alpha = solveLinSearch([=] (float lambda) {
                                                   return checker(pOld.r + dr*lambda) + insideTolerance;
                                               });

            float3 candidate = (alpha >= 0.0f) ? pOld.r + alpha * dr : pOld.r;
            candidate = rescue(candidate, dt, insideTolerance, p.i1, checker);

            float3 uWall = velField(p.r);
            float3 unew = 2*uWall - p.u;

            localForce += (p.u - unew) / dt; // force exerted by the particle on the wall

            p.r = candidate;
            p.u = unew;
                           
            p.write2Float4(view.particles, pid);
        }

        localForce = warpReduce(localForce, [](float a, float b){return a+b;});

        if ((threadIdx.x % warpSize == 0) &&
            (length(localForce) > 1e-6f))
            atomicAdd(totalForce, make_double3(localForce));
    }
}

} // namespace bounceKernels
