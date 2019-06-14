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

__device__ inline
Segment readMatFrame(const float4 *rodPos, int segmentId)
{
    return {make_float3( rodPos[5*segmentId + 1] ),
            make_float3( rodPos[5*segmentId + 2] )};
}



static constexpr float NoCollision = -1.f;

__device__ inline float squaredDistanceToSegment(const float3& r0, const float3& r1, const float3& x)
{
    float3 dr = r1 - r0;
    float alpha = dot(x - r0, dr) / dot(dr, dr);
    alpha = min(1.f, max(0.f, alpha));
    float3 p = r0 + alpha * dr;
    float3 dx = x - p;
    return dot(dx, dx);
}

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
        float3  xt = xOld +      t * dx;

        float dsq = squaredDistanceToSegment(r0t, r1t, xt);
        return dsq - radius*radius;
    };

    if (F(1.f) > 0.f) return NoCollision;

    constexpr float tol = 1e-6f;
    float alpha = solveLinSearch(F, 0.0f, 1.0f, tol);

    if (alpha >= 0.0f && alpha <= 1.0f)
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

        if (alpha == NoCollision) continue;

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

    const int3 cidLow  = cinfo.getCellIdAlongAxes(lo - (radius + tol));
    const int3 cidHigh = cinfo.getCellIdAlongAxes(hi + (radius + tol));

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



__device__ inline auto interpolate(const float3& r0, const float3& r1, float a)
{
    return a * r0 + (1.f-a) * r1;
}

__device__ inline Segment interpolate(const Segment& s0, const Segment& s1, float a)
{
    return {interpolate(s0.r0, s1.r0, a),
            interpolate(s0.r1, s1.r1, a)};
}


// compute coordinates of a point in a rod with material frame
// coords are in directions of (rod segment, material frame, cross product of the first 2)
// origin is at rod start (r0)
__device__ inline float3 getLocalCoords(float3 x, const Segment& seg, const Segment& mat)
{
    auto t = normalize(seg.r1 - seg.r0);
    auto u = mat.r1 - mat.r0;
    u = normalize(u - dot(u, t) * t);
    auto v = cross(t, u);

    x -= seg.r0;
    return {dot(x, t), dot(x, u), dot(x, v)}; 
}


__device__ inline float3 getLocalCoords(const float3& xNew, const float3& xOld,
                                        const Segment& segNew, const Segment& segOld,
                                        const Segment& matNew, const Segment& matOld,
                                        float alpha)
{
    auto colPoint = interpolate(xOld, xNew, alpha);
    auto colSeg = interpolate(segOld, segNew, alpha);
    auto colMat = interpolate(matOld, matNew, alpha);
    return getLocalCoords(colPoint, colSeg, colMat); 
}

__device__ inline float3 localToCartesianCoords(const float3& local, const Segment& seg, const Segment& mat)
{
    auto t = normalize(seg.r1 - seg.r0);
    auto u = mat.r1 - mat.r0;
    u = normalize(u - dot(u, t) * t);
    auto v = cross(t, u);

    float3 x = local.x * t + local.y * u + local.z * v;
    return seg.r0 + x;
}

struct Forces
{
    float3 fr0, fr1, fu0, fu1;
};

__device__ inline Forces transferMomentumToSegment(float dt, float partMass, const float3& pos, const float3& dV,
                                                   const Segment& seg, const Segment& mat)
{
    Forces out;
    float3 rc = 0.5f * (seg.r0 + seg.r1);
    float3 dx = pos - rc;
    
    float3 F = (partMass / dt) * dV;
    float3 T = cross(dx, F);

    // linear momentum equaly to everyone
    out.fr0 = out.fr1 = out.fu0 = out.fu1 = -0.25f * F;

    float3 dr = seg.r1 - seg.r0;
    auto t = normalize(dr);
    float3 du = mat.r1 - mat.r0;

    float3 Tpara = dot(T, t) * t;
    float3 Tperp = T - Tpara;

    float tdu = dot(du, t);
    float3 du_ = du - tdu * t;

    float paraFactor = 1.f / (dot(du, du) + tdu*tdu);
    
    float3 fTpara = (0.5f * paraFactor) * cross(du, Tpara);

    // the above force gives extra torque in Tperp direction
    // compensate that here
    Tperp -= (paraFactor * tdu*tdu * length(Tpara)) * du_;
    
    float3 fTperp = (0.5f / dot(dr, dr)) * cross(dr, Tperp);

    out.fr0 -= fTperp;
    out.fr1 += fTperp;

    out.fu0 -= fTpara;
    out.fu1 += fTpara;

    return out;
}

__global__ void performBouncing(RVviewWithOldParticles rvView, float radius,
                                PVviewWithOldParticles pvView, int nCollisions,
                                const int2 *collisionInfos, const int *collisionTimes,
                                float dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nCollisions) return;

    auto collisionInfo = collisionInfos[i];
    int pid       = collisionInfo.x;
    int globSegId = collisionInfo.y;

    const int rodId = globSegId / rvView.nSegments;
    const int segId = globSegId % rvView.nSegments;

    auto segNew = readSegment(rvView.positions    + rodId * rvView.objSize, segId);
    auto segOld = readSegment(rvView.oldPositions + rodId * rvView.objSize, segId);

    auto matNew = readMatFrame(rvView.positions    + rodId * rvView.objSize, segId);
    auto matOld = readMatFrame(rvView.oldPositions + rodId * rvView.objSize, segId);

    Particle p (pvView.readParticle(pid));
    
    auto rNew = p.r;
    auto rOld = pvView.readOldPosition(pid);

    auto alpha = collision(radius, segNew, segOld, rNew, rOld);

    // perform the collision only with the first rod encountered
    int minTime = collisionTimes[pid];
    if (1.0f - alpha != __int_as_float(minTime)) return;

    auto localCoords = getLocalCoords(rNew, rOld, segNew, segOld, matNew, matOld, alpha);

    float3 colPosNew = localToCartesianCoords(localCoords, segNew, matNew);
    float3 colPosOld = localToCartesianCoords(localCoords, segOld, matOld);
    float3 colVel    = (1.f/dt) * (colPosNew - colPosOld);

    // bounce back velocity
    float3 newVel = 2 * colVel - p.u;

    auto segF = transferMomentumToSegment(dt, pvView.mass, colPosNew, newVel - p.u, segNew, matNew);
    
    p.r = colPosNew;
    p.u = newVel;
    
    pvView.writeParticle(pid, p);

    auto faddr = rvView.forces + rodId * rvView.objSize + 5*segId;

    atomicAdd(faddr + 0, segF.fr0);
    atomicAdd(faddr + 1, segF.fu0);
    atomicAdd(faddr + 2, segF.fu1);
    atomicAdd(faddr + 5, segF.fr1);
}

    
} // namespace RodBounceKernels
