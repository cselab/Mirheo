#pragma once

#include "common.h"

#include <core/bounce_solver.h>
#include <core/celllist.h>
#include <core/pvs/views/rv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>

namespace RodBounceKernels
{

struct Segment
{
    float3 r0, r1;
};

using SegmentTable = CollisionTable<int2>;


__device__ inline
Segment readSegment(const float4 *rodPos, int segmentId)
{
    return {make_float3( rodPos[5*(segmentId + 0)] ),
            make_float3( rodPos[5*(segmentId + 1)] )};
}



static constexpr float NoCollision = -1.f;

// find "time" (0.0 to 1.0) of the segment - moving triangle intersection
// returns NoCollision is no intersection
// sets intPoint and intSegment if intersection found
__device__ inline
float collision(const float radius,
                const Segment& segNew, const Segment& segOld,
                float3 xNew, float3 xOld)
{
    const auto dx  = xNew - xOld;
    const auto dr0 = segNew.r0 - segOld.r0;
    const auto dr1 = segNew.r1 - segOld.r1;

    // Signed distance to a segment of given radius
    auto F = [=] (float t) {
        float3 r0t = segOld.r0 + t * dr0;
        float3 r1t = segOld.r1 + t * dr1;
        float3 xt = xOld +       t * dx;

        float3 e = r1t - r0t;
        float  a = dot(xt - r0t, e) / dot(e, e);
        a = min(1.f, max(0.f, a));
        float3 p = r0t + a * e;
        return length(p-xt) - radius;
    };

    constexpr float tol = 1e-6f;
    float2 res = solveLinSearch_verbose(F, 0.0f, 1.0f, tol);
    auto alpha = res.x;
    auto Fval  = res.y;

    if (fabs(Fval) < tol && alpha >= 0.0f && alpha <= 1.0f)
        return alpha;

    return NoCollision;
}

__device__ inline
void findBouncesInCell(int pstart, int pend, int globSegId,
                       const float radius,
                       const Segment& segNew, const Segment& segOld,
                       PVviewWithOldParticles pvView,
                       SegmentTable segmentTable, int *collisionTimes)
{
    #pragma unroll 2
    for (int pid = pstart; pid < pend; ++pid)
    {
        auto rNew = make_float3(pvView.readPosition(pid));
        auto rOld = pvView.readOldPosition(pid);

        auto alpha = collision(radius, segNew, segOld, rNew, rOld);

        if (alpha == NoCollision) return;

        atomicMax(collisionTimes+pid, __float_as_int(1.0f - alpha));
        segmentTable.push_back({pid, globSegId});
    }
}

__global__ void findBounces(RVviewWithOldParticles rvView, float radius,
                            PVviewWithOldParticles pvView, CellListInfo cinfo,
                            SegmentTable segmentTable, int *collisionTimes)
{
    // About maximum distance a particle can cover in one step
    const float tol = 0.25f;

    // One thread per segment
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int rodId = gid / rvView.nSegments;
    const int segId = gid % rvView.nSegments;
    if (rodId >= rvView.nObjects) return;

    auto segNew = readSegment(rvView.positions    + rodId * rvView.objSize, segId);
    auto segOld = readSegment(rvView.oldPositions + rodId * rvView.objSize, segId);

    const float3 lo = fmin_vec(segNew.r0, segNew.r1, segOld.r0, segOld.r1);
    const float3 hi = fmax_vec(segNew.r0, segNew.r1, segOld.r0, segOld.r1);

    const int3 cidLow  = cinfo.getCellIdAlongAxes(lo - tol);
    const int3 cidHigh = cinfo.getCellIdAlongAxes(hi + tol);

        int3 cid3;

    #pragma unroll 2
    for (cid3.z = cidLow.z; cid3.z <= cidHigh.z; ++cid3.z)
    {
        for (cid3.y = cidLow.y; cid3.y <= cidHigh.y; ++cid3.y)
        {
            cid3.x = cidLow.x;
            int cidLo = max(cinfo.encode(cid3), 0);
            
            cid3.x = cidHigh.x;
            int cidHi = min(cinfo.encode(cid3)+1, cinfo.totcells);
            
            int pstart = cinfo.cellStarts[cidLo];
            int pend   = cinfo.cellStarts[cidHi];
            
            findBouncesInCell(pstart, pend, gid, radius,
                              segNew, segOld, pvView,
                              segmentTable, collisionTimes);
        }
    }
}

} // namespace RodBounceKernels
