#pragma once

#include "common.h"

#include <core/bounce_solver.h>
#include <core/celllist.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>

namespace MeshBounceKernels
{

struct Triangle
{
    float3 v0, v1, v2;
};

using TriangleTable = CollisionTable<int2>;


__device__ inline
Triangle readTriangle(const float4 *vertices, int startId, int3 trid)
{
    auto addr = vertices + startId;
    return {
        make_float3( addr[trid.x] ),
        make_float3( addr[trid.y] ),
        make_float3( addr[trid.z] ) };
}



__device__ inline
bool segmentTriangleQuickCheck(Triangle trNew, Triangle trOld,
                               float3 xNew, float3 xOld)
{
    const float3 v0 = trOld.v0;
    const float3 v1 = trOld.v1;
    const float3 v2 = trOld.v2;

    const float3 dx  = xNew - xOld;
    const float3 dv0 = trNew.v0 - v0;
    const float3 dv1 = trNew.v1 - v1;
    const float3 dv2 = trNew.v2 - v2;

    // Distance to the triangle plane
    auto F = [=] (float t) {
        float3 v0t = v0 + t*dv0;
        float3 v1t = v1 + t*dv1;
        float3 v2t = v2 + t*dv2;

        float3 nt = normalize(cross(v1t-v0t, v2t-v0t));
        float3 xt = xOld + t*dx;
        return  dot( xt - v0t, nt );
    };

    // d / dt (non normalized Distance)
    auto F_prime = [=] (float t) {
        float3 v0t = v0 + t*dv0;
        float3 v1t = v1 + t*dv1;
        float3 v2t = v2 + t*dv2;

        float3 nt = cross(v1t-v0t, v2t-v0t);

        float3 xt = xOld + t*dx;
        return dot(dx-dv0, nt) + dot(xt-v0t, cross(dv1-dv0, v2t-v0t) + cross(v1t-v0t, dv2-dv0));
    };

    auto F0 = F(0.0f);
    auto F1 = F(1.0f);

    // assume that particles don t move more than this distance every time step
    const float tolDistance = 0.1;
    
    if (fabs(F0) > tolDistance && fabs(F1) > tolDistance)
        return false;
    
    if (F0 * F1 < 0.0f)
        return true;

    // XXX: This is not always correct
    if (F_prime(0.0f) * F_prime(1.0f) >= 0.0f)
        return false;

    return true;
}

__device__ inline
void findBouncesInCell(int pstart, int pend, int globTrid,
                       Triangle tr, Triangle trOld,
                       PVviewWithOldParticles pvView,
                       MeshView mesh,
                       TriangleTable triangleTable)
{

#pragma unroll 2
    for (int pid = pstart; pid < pend; pid++)
    {
        Particle p;
        pvView.readPosition   (p,    pid);
        auto rOld = pvView.readOldPosition(pid);

        if (segmentTriangleQuickCheck(tr, trOld, p.r, rOld))
            triangleTable.push_back({pid, globTrid});
    }
}

//__launch_bounds__(128, 6)
static __global__
void findBouncesInMesh(OVviewWithNewOldVertices objView,
                       PVviewWithOldParticles pvView,
                       MeshView mesh,
                       CellListInfo cinfo,
                       TriangleTable triangleTable)
{
    // About maximum distance a particle can cover in one step
    const float tol = 0.2f;

    // One THREAD per triangle
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int objId = gid / mesh.ntriangles;
    const int trid  = gid % mesh.ntriangles;
    if (objId >= objView.nObjects) return;

    const int3 triangle = mesh.triangles[trid];
    Triangle tr =    readTriangle(objView.vertices    , mesh.nvertices*objId, triangle);
    Triangle trOld = readTriangle(objView.old_vertices, mesh.nvertices*objId, triangle);

    const float3 lo = fmin_vec(trOld.v0, trOld.v1, trOld.v2, tr.v0, tr.v1, tr.v2);
    const float3 hi = fmax_vec(trOld.v0, trOld.v1, trOld.v2, tr.v0, tr.v1, tr.v2);

    const int3 cidLow  = cinfo.getCellIdAlongAxes(lo - tol);
    const int3 cidHigh = cinfo.getCellIdAlongAxes(hi + tol);

    int3 cid3;
#pragma unroll 2
    for (cid3.z = cidLow.z; cid3.z <= cidHigh.z; cid3.z++)
        for (cid3.y = cidLow.y; cid3.y <= cidHigh.y; cid3.y++)
            {
                cid3.x = cidLow.x;
                int cidLo = max(cinfo.encode(cid3), 0);

                cid3.x = cidHigh.x;
                int cidHi = min(cinfo.encode(cid3)+1, cinfo.totcells);

                int pstart = cinfo.cellStarts[cidLo];
                int pend   = cinfo.cellStarts[cidHi];

                findBouncesInCell(pstart, pend, gid, tr, trOld, pvView, mesh, triangleTable);
            }
}

//=================================================================================================================
// Filter the collisions better
//=================================================================================================================


__device__ inline bool isInside(Triangle tr, float3 p)
{
    const float edgeTolerance = 1e-18f;

    auto signedArea2 = [] (float3 a, float3 b, float3 c, float3 direction) {
        auto n = cross(a-b, a-c);
        auto sign = dot(n, direction);

        auto S2 = dot(n, n);
        return (sign >= 0.0f) ? S2 : -S2;
    };

    float3 n = cross(tr.v1-tr.v0, tr.v2-tr.v0);

    float s0 = signedArea2(tr.v0, tr.v1, p, n);
    float s1 = signedArea2(tr.v1, tr.v2, p, n);
    float s2 = signedArea2(tr.v2, tr.v0, p, n);


    return (s0 > -edgeTolerance && s1 > -edgeTolerance && s2 > -edgeTolerance);
}


__device__ inline void sort3(float2* v)
{
    auto swap = [] (float2& a, float2& b) {
        float2 tmp = a;
        a = b;
        b = tmp;
    };

    if (v[0].x > v[1].x) swap(v[0], v[1]);
    if (v[0].x > v[2].x) swap(v[0], v[2]);
    if (v[1].x > v[2].x) swap(v[1], v[2]);
}

// find "time" (0.0 to 1.0) of the segment - moving triangle intersection
// returns -1 is no intersection
// sets intPoint and intTriangle if intersection found
__device__ inline float
intersectSegmentWithTriangle(Triangle trNew, Triangle trOld,
                             float3 xNew, float3 xOld,
                             float3& intPoint,
                             Triangle& intTriangle,
                             float& intSign,
                             int trid = -1)
{
    const float tol = 2e-6f;

    const float3 v0 = trOld.v0;
    const float3 v1 = trOld.v1;
    const float3 v2 = trOld.v2;

    const float3 dx  = xNew - xOld;
    const float3 dv0 = trNew.v0 - v0;
    const float3 dv1 = trNew.v1 - v1;
    const float3 dv2 = trNew.v2 - v2;


    // precompute scaling factor
    auto n = cross(trNew.v1-trNew.v0,
                   trNew.v2-trNew.v0);
    float n_1 = rsqrtf(dot(n, n));

    // Distance to a triangle
    auto F = [=] (float t) {
        float3 v0t = v0 + t*dv0;
        float3 v1t = v1 + t*dv1;
        float3 v2t = v2 + t*dv2;

        float3 xt = xOld + t*dx;
        return  n_1 * dot( xt - v0t, cross(v1t-v0t, v2t-v0t) );
    };

    // d / dt (Distance)
    auto F_prime = [=] (float t) {
        float3 v0t = v0 + t*dv0;
        float3 v1t = v1 + t*dv1;
        float3 v2t = v2 + t*dv2;

        float3 nt = cross(v1t-v0t, v2t-v0t);

        float3 xt = xOld + t*dx;
        return  n_1 * ( dot(dx-dv0, nt) + dot(xt-v0t, cross(dv1-dv0, v2t-v0t) + cross(v1t-v0t, dv2-dv0)) );
    };

    // Has side-effects!!
    auto checkIfInside = [&] (float alpha) {
        intPoint = xOld + alpha*dx;

        intTriangle.v0 = v0 + alpha*dv0;
        intTriangle.v1 = v1 + alpha*dv1;
        intTriangle.v2 = v2 + alpha*dv2;

        intSign = -F_prime(alpha);

        return isInside(intTriangle, intPoint);
    };

    float2 roots[3];
    roots[0] = solveNewton(F, F_prime, 0.0f);
    roots[2] = solveNewton(F, F_prime, 1.0f);


    float left, right;

    if (F(0.0f)*F(1.0f) < 0.0f)
    {
        // Three roots
        if (roots[0].x >= 0.0f && roots[0].x <= 1.0f && fabsf(roots[0].y) < tol &&
            roots[2].x >= 0.0f && roots[2].x <= 1.0f && fabsf(roots[2].y) < tol)
        {
            left  = roots[0].x + 1e-5f/fabsf(F_prime(roots[0].x));
            right = roots[2].x - 1e-5f/fabsf(F_prime(roots[2].x));
        }
        else // One root
        {
            left  = 0.0f;
            right = 1.0f;
        }
    }
    else  // Maybe two roots
    {
        float2 newtonRoot;

        if (roots[0].x >= 0.0f && roots[0].x <= 1.0f && fabsf(roots[0].y) < tol)
            newtonRoot = roots[0];

        if (roots[2].x >= 0.0f && roots[2].x <= 1.0f && fabsf(roots[2].y) < tol)
            newtonRoot = roots[2];

        if (F(0.0f) * F_prime(newtonRoot.x) > 0.0f)
        {
            left  = 0.0f;
            right = newtonRoot.x - 1e-5f/fabsf(F_prime(newtonRoot.x));
        }
        else
        {
            left  = newtonRoot.x + 1e-5f/fabsf(F_prime(newtonRoot.x));
            right = 1.0f;
        }
    }

    roots[1] = solveLinSearch_verbose(F, left, right);

    sort3(roots);

    if ( fabs(roots[0].y) < tol && roots[0].x >= 0.0f && roots[0].x <= 1.0f )
        if (checkIfInside(roots[0].x)) return roots[0].x;

    if ( fabs(roots[1].y) < tol && roots[1].x >= 0.0f && roots[1].x <= 1.0f )
        if (checkIfInside(roots[1].x)) return roots[1].x;

    if ( fabs(roots[2].y) < tol && roots[2].x >= 0.0f && roots[2].x <= 1.0f )
        if (checkIfInside(roots[2].x)) return roots[2].x;

    return -1.0f;
}

static __global__
void refineCollisions(OVviewWithNewOldVertices objView,
                      PVviewWithOldParticles pvView,
                      MeshView mesh,
                      int nCoarseCollisions, int2 *coarseTable,
                      TriangleTable fineTable,
                      int *collisionTimes)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nCoarseCollisions) return;

    const int2 pid_trid = coarseTable[gid];
    int pid = pid_trid.x;

    Particle p (pvView.readParticle   (pid));
    auto rOld = pvView.readOldPosition(pid);

    const int trid  = pid_trid.y % mesh.ntriangles;
    const int objId = pid_trid.y / mesh.ntriangles;

    const int3 triangle = mesh.triangles[trid];
    Triangle tr =    readTriangle(objView.vertices    , mesh.nvertices*objId, triangle);
    Triangle trOld = readTriangle(objView.old_vertices, mesh.nvertices*objId, triangle);

    float3 intPoint;
    Triangle intTriangle;
    float intSign;
    float alpha = intersectSegmentWithTriangle(tr, trOld, p.r, rOld, intPoint, intTriangle, intSign);

    if (alpha < -0.1f) return;

    atomicMax(collisionTimes+pid, __float_as_int(1.0f - alpha));
    fineTable.push_back(pid_trid);
}



//=================================================================================================================
// Perform the damn collisions
//=================================================================================================================


// p is assumed to be in the a-b-c plane
// a lot more precise method that the one solving a linear system
__device__ inline
float3 barycentric(Triangle tr, float3 p)
{
    auto signedArea = [] (float3 a, float3 b, float3 c, float3 direction) {
        auto n = cross(a-b, a-c);
        auto sign = dot(n, direction);

        auto S = length(n);
        return (sign >= 0.0f) ? S : -S;
    };

    auto n = cross(tr.v0-tr.v1, tr.v0-tr.v2);
    auto s0_1 = rsqrtf(dot(n, n));

    auto s1 = signedArea(tr.v0, tr.v1, p, n);
    auto s2 = signedArea(tr.v1, tr.v2, p, n);
    auto s3 = signedArea(tr.v2, tr.v0, p, n);

    return make_float3(s2, s3, s1) * s0_1;
}

/**
 * Reflect the velocity, in the triangle's reference frame
 */
__device__ inline
float3 reflectVelocity(float3 n, float kbT, float mass, float seed1, float seed2)
{
    const int maxTries = 50;
    // reflection with random scattering
    // according to Maxwell distr
    float2 rand1 = Saru::normal2(seed1, threadIdx.x, blockIdx.x);
    float2 rand2 = Saru::normal2(seed2, threadIdx.x, blockIdx.x);

    float3 r = make_float3(rand1.x, rand1.y, rand2.x);
    for (int i=0; i<maxTries; i++)
    {
        if (dot(r, n) > 0) break;

        float2 rand3 = Saru::normal2(rand2.y, threadIdx.x, blockIdx.x);
        float2 rand4 = Saru::normal2(rand3.y, threadIdx.x, blockIdx.x);
        r = make_float3(rand3.x, rand3.y, rand4.x);
    }
    r = normalize(r) * sqrtf(kbT / mass);

    return r;
}


// Particle with mass M and velocity U0 hits triangle tr (v0, v1, v2)
// into point O. Its new velocity is Unew.
// Vertex masses are m. Treated as rigid and stationary,
// what are the vertex forces induced by the collision?
__device__ inline
void triangleForces(Triangle tr, float m,
                    float3 O_barycentric, float3 U0, float3 Unew, float M,
                    float dt,
                    float3& f0, float3& f1, float3& f2)
{
    const float tol = 1e-5f;

    auto len2 = [] (float3 x) {
        return dot(x, x);
    };

    const float3 n = normalize(cross(tr.v1-tr.v0, tr.v2-tr.v0));

    const float3 dU = U0 - Unew;
    const float IU_ortI = dot(dU, n);
    const float3 U_par = dU - IU_ortI * n;

    const float a = M * IU_ortI;
    const float v0_ort = O_barycentric.x * a;
    const float v1_ort = O_barycentric.y * a;
    const float v2_ort = O_barycentric.z * a;

    const float3 C = 0.333333333f * (tr.v0+tr.v1+tr.v2);
    const float3 Vc = 0.333333333f * M * U_par;

    const float3 O = O_barycentric.x * tr.v0 + O_barycentric.y * tr.v1 + O_barycentric.z * tr.v2;
    const float3 L = M * cross(C-O, U_par);

    const float J = len2(C-tr.v0) + len2(C-tr.v1) + len2(C-tr.v2);
    if (fabs(J) < tol)
    {
        float3 f = dU * M / dt;
        f0 = O_barycentric.x*f;
        f1 = O_barycentric.y*f;
        f2 = O_barycentric.z*f;

        return;
    }

    const float w = -dot(L, n) / J;

    const float3 orth_r0 = cross(C-tr.v0, n);
    const float3 orth_r1 = cross(C-tr.v1, n);
    const float3 orth_r2 = cross(C-tr.v2, n);

    const float3 u0 = w * orth_r0;
    const float3 u1 = w * orth_r1;
    const float3 u2 = w * orth_r2;

    const float3 v0 = v0_ort*n + Vc + u0;
    const float3 v1 = v1_ort*n + Vc + u1;
    const float3 v2 = v2_ort*n + Vc + u2;

    const float invdt = 1.0f / dt;
    f0 = v0 * invdt;
    f1 = v1 * invdt;
    f2 = v2 * invdt;
}


static __global__
void performBouncingTriangle(OVviewWithNewOldVertices objView,
                             PVviewWithOldParticles pvView,
                             MeshView mesh,
                             int nCollisions, int2 *collisionTable, int *collisionTimes,
                             const float dt,
                             float kbT, float seed1, float seed2)
{
    const float eps = 5e-5f;

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nCollisions) return;

    const int2 pid_trid = collisionTable[gid];
    int pid = pid_trid.x;

    Particle p (pvView.readParticle   (pid));
    auto rOld = pvView.readOldPosition(pid);
    Particle corrP = p;

    float3 f0, f1, f2;

    const int trid  = pid_trid.y % mesh.ntriangles;
    const int objId = pid_trid.y / mesh.ntriangles;

    const int3 triangle = mesh.triangles[trid];
    Triangle tr =    readTriangle(objView.vertices    , mesh.nvertices*objId, triangle);
    Triangle trOld = readTriangle(objView.old_vertices, mesh.nvertices*objId, triangle);

    float3 intPoint;
    Triangle intTriangle;
    float intSign;
    float alpha = intersectSegmentWithTriangle(tr, trOld, p.r, rOld, intPoint, intTriangle, intSign);

    int minTime = collisionTimes[pid];

    if (1.0f - alpha != __int_as_float(minTime)) return;

    float3 barycentricCoo = barycentric(intTriangle, intPoint);

    const float dt_1 = 1.0f / dt;
    Triangle trVel = { (tr.v0-trOld.v0)*dt_1, (tr.v1-trOld.v1)*dt_1, (tr.v2-trOld.v2)*dt_1 };

    // Position is based on INTERMEDIATE barycentric collision coordinates and FINAL triangle
    const float3 vtri = barycentricCoo.x*trVel.v0 + barycentricCoo.y*trVel.v1 + barycentricCoo.z*trVel.v2;
    const float3 coo  = barycentricCoo.x*tr.v0    + barycentricCoo.y*tr.v1    + barycentricCoo.z*tr.v2;

    const float3 n = normalize(cross(tr.v1-tr.v0, tr.v2-tr.v0));

    // new velocity relative to the triangle speed
    float3 newV = reflectVelocity( (intSign > 0) ? n : -n, kbT, pvView.mass, seed1, seed2 );

    triangleForces(tr, objView.mass, barycentricCoo, p.u - vtri, newV, pvView.mass, dt, f0, f1, f2);

    corrP.r = coo + eps * ((intSign > 0) ? n : -n);
    corrP.u = newV + vtri;

    float sign = dot( corrP.r-tr.v0, cross(tr.v1-tr.v0, tr.v2-tr.v0) );


    pvView.writeParticle(pid, corrP);

    atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.x, f0);
    atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.y, f1);
    atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.z, f2);
}

} // namespace MeshBounceKernels
