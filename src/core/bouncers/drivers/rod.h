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
    real3 r0, r1;
};

using SegmentTable = CollisionTable<int2>;


__device__ static inline
Segment readSegment(const real4 *rodPos, int segmentId)
{
    return {make_real3( rodPos[5*(segmentId + 0)] ),
            make_real3( rodPos[5*(segmentId + 1)] )};
}

__device__ static inline
Segment readMatFrame(const real4 *rodPos, int segmentId)
{
    return {make_real3( rodPos[5*segmentId + 1] ),
            make_real3( rodPos[5*segmentId + 2] )};
}



static constexpr real NoCollision = -1._r;

__device__ static inline real3 projectionPointOnSegment(const Segment& s, const real3& x)
{
    const real3 dr = s.r1 - s.r0;
    real alpha = dot(x - s.r0, dr) / dot(dr, dr);
    alpha = math::min(1._r, math::max(0._r, alpha));
    const real3 p = s.r0 + alpha * dr;
    return p;
}


__device__ static inline real squaredDistanceToSegment(const Segment& s, const real3& x)
{
    const real3 dx = x - projectionPointOnSegment(s, x);
    return dot(dx, dx);
}

__device__ static inline real3 segmentNormal(const Segment& s, const real3& x)
{
    const real3 dx = x - projectionPointOnSegment(s, x);
    return normalize(dx);
}

// find "time" (0.0 to 1.0) of the segment - moving triangle intersection
// returns NoCollision is no intersection
// sets intPoint and intSegment if intersection found
__device__ static inline
real collision(const real radius,
                const Segment& segNew, const Segment& segOld,
                real3 xNew, real3 xOld)
{
    const auto dx  = xNew - xOld;
    const auto dr0 = segNew.r0 - segOld.r0;
    const auto dr1 = segNew.r1 - segOld.r1;

    // Signed distance to a segment of given radius
    auto F = [=] (real t)
    {
        const Segment st = {segOld.r0 + t * dr0,
                            segOld.r1 + t * dr1};
        const real3  xt =  xOld +      t * dx;

        const real dsq = squaredDistanceToSegment(st, xt);
        return dsq - radius*radius;
    };

    constexpr RootFinder::Bounds limits {0._r, 1._r};
    
    if (F(limits.up) > 0._r) return NoCollision;

    constexpr real tol = 1e-6_r;
    const real alpha = RootFinder::linearSearch(F, limits, tol);

    if (alpha >= limits.lo && alpha <= limits.up)
        return alpha;

    return NoCollision;
}

__device__ static inline
void findBouncesInCell(int pstart, int pend, int globSegId,
                       const real radius,
                       const Segment& segNew, const Segment& segOld,
                       PVviewWithOldParticles pvView,
                       SegmentTable segmentTable, int *collisionTimes)
{
    #pragma unroll 2
    for (int pid = pstart; pid < pend; ++pid)
    {
        const real3 rNew = make_real3(pvView.readPosition(pid));
        const real3 rOld = pvView.readOldPosition(pid);

        const auto alpha = collision(radius, segNew, segOld, rNew, rOld);

        if (alpha == NoCollision) continue;

        atomicMax(collisionTimes+pid, __float_as_int(static_cast<float>(1.0_r - alpha)));
        segmentTable.push_back({pid, globSegId});
    }
}

__global__ void findBounces(RVviewWithOldParticles rvView, real radius,
                            PVviewWithOldParticles pvView, CellListInfo cinfo,
                            SegmentTable segmentTable, int *collisionTimes)
{
    // About maximum distance a particle can cover in one step
    constexpr real tol = 0.25_r;

    // One thread per segment
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int rodId = gid / rvView.nSegments;
    const int segId = gid % rvView.nSegments;
    if (rodId >= rvView.nObjects) return;

    auto segNew = readSegment(rvView.positions    + rodId * rvView.objSize, segId);
    auto segOld = readSegment(rvView.oldPositions + rodId * rvView.objSize, segId);

    const real3 lo = fmin_vec(segNew.r0, segNew.r1, segOld.r0, segOld.r1);
    const real3 hi = fmax_vec(segNew.r0, segNew.r1, segOld.r0, segOld.r1);

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
            const int cidHi = math::min(cinfo.encode(cid3)+1, cinfo.totcells);
            
            const int pstart = cinfo.cellStarts[cidLo];
            const int pend   = cinfo.cellStarts[cidHi];
            
            findBouncesInCell(pstart, pend, gid, radius,
                              segNew, segOld, pvView,
                              segmentTable, collisionTimes);
        }
    }
}



__device__ static inline auto interpolate(const real3& r0, const real3& r1, real a)
{
    return a * r0 + (1._r-a) * r1;
}

__device__ static inline Segment interpolate(const Segment& s0, const Segment& s1, real a)
{
    return {interpolate(s0.r0, s1.r0, a),
            interpolate(s0.r1, s1.r1, a)};
}


// compute coordinates of a point in a rod with material frame
// coords are in directions of (rod segment, material frame, cross product of the first 2)
// origin is at rod start (r0)
__device__ static inline real3 getLocalCoords(real3 x, const Segment& seg, const Segment& mat)
{
    auto t = normalize(seg.r1 - seg.r0);
    auto u = mat.r1 - mat.r0;
    u = normalize(u - dot(u, t) * t);
    auto v = cross(t, u);

    x -= seg.r0;
    return {dot(x, t), dot(x, u), dot(x, v)}; 
}


__device__ static inline real3 getLocalCoords(const real3& xNew, const real3& xOld,
                                               const Segment& segNew, const Segment& segOld,
                                               const Segment& matNew, const Segment& matOld,
                                               real alpha)
{
    const auto colPoint = interpolate(xOld, xNew, alpha);
    const auto colSeg = interpolate(segOld, segNew, alpha);
    const auto colMat = interpolate(matOld, matNew, alpha);
    return getLocalCoords(colPoint, colSeg, colMat); 
}

__device__ static inline real3 localToCartesianCoords(const real3& local, const Segment& seg, const Segment& mat)
{
    const real3 t = normalize(seg.r1 - seg.r0);
    real3 u = mat.r1 - mat.r0;
    u = normalize(u - dot(u, t) * t);
    const real3 v = cross(t, u);

    const real3 x = local.x * t + local.y * u + local.z * v;
    return seg.r0 + x;
}

struct Forces
{
    real3 fr0, fr1, fu0, fu1;
};

__device__ static inline Forces transferMomentumToSegment(real dt, real partMass, const real3& pos, const real3& dV,
                                                          const Segment& seg, const Segment& mat)
{
    Forces out;
    const real3 rc = 0.5_r * (seg.r0 + seg.r1);
    const real3 dx = pos - rc;
    
    const real3 F = (partMass / dt) * dV;
    const real3 T = cross(dx, F);

    // linear momentum equaly to everyone
    out.fr0 = out.fr1 = out.fu0 = out.fu1 = -0.25_r * F;

    const real3 dr = seg.r1 - seg.r0;
    const auto t = normalize(dr);
    const real3 du = mat.r1 - mat.r0;

    const real3 Tpara = dot(T, t) * t;
    real3 Tperp = T - Tpara;

    const real tdu = dot(du, t);
    const real3 du_ = du - tdu * t;

    const real paraFactor = 1._r / (dot(du, du) + tdu*tdu);
    
    const real3 fTpara = (0.5_r * paraFactor) * cross(du, Tpara);

    // the above force gives extra torque in Tperp direction
    // compensate that here
    Tperp -= (paraFactor * tdu*tdu * length(Tpara)) * du_;
    
    const real3 fTperp = (0.5_r / dot(dr, dr)) * cross(dr, Tperp);

    out.fr0 -= fTperp;
    out.fr1 += fTperp;

    out.fu0 -= fTpara;
    out.fu1 += fTpara;

    return out;
}

template <class BounceKernel>
__global__ void performBouncing(RVviewWithOldParticles rvView, real radius,
                                PVviewWithOldParticles pvView, int nCollisions,
                                const int2 *collisionInfos, const int *collisionTimes,
                                real dt, const BounceKernel bounceKernel)
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
    
    const real3 rNew = p.r;
    const real3 rOld = pvView.readOldPosition(pid);

    const real alpha = collision(radius, segNew, segOld, rNew, rOld);

    // perform the collision only with the first rod encountered
    const int minTime = collisionTimes[pid];
    if (static_cast<float>(1.0_r - alpha) != __int_as_float(minTime)) return;

    const auto localCoords = getLocalCoords(rNew, rOld, segNew, segOld, matNew, matOld, alpha);

    const real3 colPosNew = localToCartesianCoords(localCoords, segNew, matNew);
    const real3 colPosOld = localToCartesianCoords(localCoords, segOld, matOld);
    const real3 colVel    = (1._r/dt) * (colPosNew - colPosOld);

    const real3 normal = segmentNormal(segNew, colPosNew);
    const real3 newVel = bounceKernel.newVelocity(p.u, colVel, normal, pvView.mass);

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
