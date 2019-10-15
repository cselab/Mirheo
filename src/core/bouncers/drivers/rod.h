#pragma once

#include "common.h"

#include <core/celllist.h>
#include <core/pvs/views/rv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/root_finder.h>

namespace RodBounceKernels
{

struct Segment
{
    float3 r0, r1;
};

using SegmentTable = CollisionTable<int2>;


__device__ static inline
Segment readSegment(const float4 *rodPos, int segmentId)
{
    return {make_float3( rodPos[5*(segmentId + 0)] ),
            make_float3( rodPos[5*(segmentId + 1)] )};
}

__device__ static inline
Segment readMatFrame(const float4 *rodPos, int segmentId)
{
    return {make_float3( rodPos[5*segmentId + 1] ),
            make_float3( rodPos[5*segmentId + 2] )};
}



static constexpr float NoCollision = -1.f;

__device__ static inline float3 projectionPointOnSegment(const Segment& s, const float3& x)
{
    const float3 dr = s.r1 - s.r0;
    float alpha = dot(x - s.r0, dr) / dot(dr, dr);
    alpha = min(1.f, max(0.f, alpha));
    const float3 p = s.r0 + alpha * dr;
    return p;
}


__device__ static inline float squaredDistanceToSegment(const Segment& s, const float3& x)
{
    const float3 dx = x - projectionPointOnSegment(s, x);
    return dot(dx, dx);
}

__device__ static inline float3 segmentNormal(const Segment& s, const float3& x)
{
    const float3 dx = x - projectionPointOnSegment(s, x);
    return normalize(dx);
}

// find "time" (0.0 to 1.0) of the segment - moving triangle intersection
// returns NoCollision is no intersection
// sets intPoint and intSegment if intersection found
__device__ static inline
float collision(const float radius,
                const Segment& segNew, const Segment& segOld,
                float3 xNew, float3 xOld)
{
    const auto dx  = xNew - xOld;
    const auto dr0 = segNew.r0 - segOld.r0;
    const auto dr1 = segNew.r1 - segOld.r1;

    // Signed distance to a segment of given radius
    auto F = [=] (float t)
    {
        const Segment st = {segOld.r0 + t * dr0,
                            segOld.r1 + t * dr1};
        const float3  xt =  xOld +      t * dx;

        const float dsq = squaredDistanceToSegment(st, xt);
        return dsq - radius*radius;
    };

    constexpr RootFinder::Bounds limits {0.f, 1.f};
    
    if (F(limits.up) > 0.f) return NoCollision;

    constexpr float tol = 1e-6f;
    const float alpha = RootFinder::linearSearch(F, limits, tol);

    if (alpha >= limits.lo && alpha <= limits.up)
        return alpha;

    return NoCollision;
}

__device__ static inline
void findBouncesInCell(int pstart, int pend, int globSegId,
                       const float radius,
                       const Segment& segNew, const Segment& segOld,
                       PVviewWithOldParticles pvView,
                       SegmentTable segmentTable, int *collisionTimes)
{
    #pragma unroll 2
    for (int pid = pstart; pid < pend; ++pid)
    {
        const float3 rNew = make_float3(pvView.readPosition(pid));
        const float3 rOld = pvView.readOldPosition(pid);

        const auto alpha = collision(radius, segNew, segOld, rNew, rOld);

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
    constexpr float tol = 0.25f;

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
            const int cidLo = max(cinfo.encode(cid3), 0);
            
            cid3.x = cidHigh.x;
            const int cidHi = min(cinfo.encode(cid3)+1, cinfo.totcells);
            
            const int pstart = cinfo.cellStarts[cidLo];
            const int pend   = cinfo.cellStarts[cidHi];
            
            findBouncesInCell(pstart, pend, gid, radius,
                              segNew, segOld, pvView,
                              segmentTable, collisionTimes);
        }
    }
}



__device__ static inline auto interpolate(const float3& r0, const float3& r1, float a)
{
    return a * r0 + (1.f-a) * r1;
}

__device__ static inline Segment interpolate(const Segment& s0, const Segment& s1, float a)
{
    return {interpolate(s0.r0, s1.r0, a),
            interpolate(s0.r1, s1.r1, a)};
}


// compute coordinates of a point in a rod with material frame
// coords are in directions of (rod segment, material frame, cross product of the first 2)
// origin is at rod start (r0)
__device__ static inline float3 getLocalCoords(float3 x, const Segment& seg, const Segment& mat)
{
    auto t = normalize(seg.r1 - seg.r0);
    auto u = mat.r1 - mat.r0;
    u = normalize(u - dot(u, t) * t);
    auto v = cross(t, u);

    x -= seg.r0;
    return {dot(x, t), dot(x, u), dot(x, v)}; 
}


__device__ static inline float3 getLocalCoords(const float3& xNew, const float3& xOld,
                                               const Segment& segNew, const Segment& segOld,
                                               const Segment& matNew, const Segment& matOld,
                                               float alpha)
{
    const auto colPoint = interpolate(xOld, xNew, alpha);
    const auto colSeg = interpolate(segOld, segNew, alpha);
    const auto colMat = interpolate(matOld, matNew, alpha);
    return getLocalCoords(colPoint, colSeg, colMat); 
}

__device__ static inline float3 localToCartesianCoords(const float3& local, const Segment& seg, const Segment& mat)
{
    const float3 t = normalize(seg.r1 - seg.r0);
    float3 u = mat.r1 - mat.r0;
    u = normalize(u - dot(u, t) * t);
    const float3 v = cross(t, u);

    const float3 x = local.x * t + local.y * u + local.z * v;
    return seg.r0 + x;
}

struct Forces
{
    float3 fr0, fr1, fu0, fu1;
};

__device__ static inline Forces transferMomentumToSegment(float dt, float partMass, const float3& pos, const float3& dV,
                                                          const Segment& seg, const Segment& mat)
{
    Forces out;
    const float3 rc = 0.5f * (seg.r0 + seg.r1);
    const float3 dx = pos - rc;
    
    const float3 F = (partMass / dt) * dV;
    const float3 T = cross(dx, F);

    // linear momentum equaly to everyone
    out.fr0 = out.fr1 = out.fu0 = out.fu1 = -0.25f * F;

    const float3 dr = seg.r1 - seg.r0;
    const auto t = normalize(dr);
    const float3 du = mat.r1 - mat.r0;

    const float3 Tpara = dot(T, t) * t;
    float3 Tperp = T - Tpara;

    const float tdu = dot(du, t);
    const float3 du_ = du - tdu * t;

    const float paraFactor = 1.f / (dot(du, du) + tdu*tdu);
    
    const float3 fTpara = (0.5f * paraFactor) * cross(du, Tpara);

    // the above force gives extra torque in Tperp direction
    // compensate that here
    Tperp -= (paraFactor * tdu*tdu * length(Tpara)) * du_;
    
    const float3 fTperp = (0.5f / dot(dr, dr)) * cross(dr, Tperp);

    out.fr0 -= fTperp;
    out.fr1 += fTperp;

    out.fu0 -= fTpara;
    out.fu1 += fTpara;

    return out;
}

template <class BounceKernel>
__global__ void performBouncing(RVviewWithOldParticles rvView, float radius,
                                PVviewWithOldParticles pvView, int nCollisions,
                                const int2 *collisionInfos, const int *collisionTimes,
                                float dt, const BounceKernel bounceKernel)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nCollisions) return;

    const auto collisionInfo = collisionInfos[i];
    const int pid       = collisionInfo.x;
    const int globSegId = collisionInfo.y;

    const int rodId = globSegId / rvView.nSegments;
    const int segId = globSegId % rvView.nSegments;

    const Segment segNew = readSegment(rvView.positions    + rodId * rvView.objSize, segId);
    const Segment segOld = readSegment(rvView.oldPositions + rodId * rvView.objSize, segId);

    const Segment matNew = readMatFrame(rvView.positions    + rodId * rvView.objSize, segId);
    const Segment matOld = readMatFrame(rvView.oldPositions + rodId * rvView.objSize, segId);

    Particle p (pvView.readParticle(pid));
    
    const float3 rNew = p.r;
    const float3 rOld = pvView.readOldPosition(pid);

    const float alpha = collision(radius, segNew, segOld, rNew, rOld);

    // perform the collision only with the first rod encountered
    const int minTime = collisionTimes[pid];
    if (1.0f - alpha != __int_as_float(minTime)) return;

    const auto localCoords = getLocalCoords(rNew, rOld, segNew, segOld, matNew, matOld, alpha);

    const float3 colPosNew = localToCartesianCoords(localCoords, segNew, matNew);
    const float3 colPosOld = localToCartesianCoords(localCoords, segOld, matOld);
    const float3 colVel    = (1.f/dt) * (colPosNew - colPosOld);

    const float3 normal = segmentNormal(segNew, colPosNew);
    const float3 newVel = bounceKernel.newVelocity(p.u, colVel, normal, pvView.mass);

    const auto segF = transferMomentumToSegment(dt, pvView.mass, colPosNew, newVel - p.u, segNew, matNew);
    
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
