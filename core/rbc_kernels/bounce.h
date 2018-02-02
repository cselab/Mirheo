#pragma once

#include <core/pvs/rbc_vector.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/bounce_solver.h>

#include <core/utils/cuda_rng.h>

struct Triangle
{
	float3 v0, v1, v2;
};

struct Intersection
{
	float3 p;
	Triangle tr;
	float alpha;
};

struct InsideTriangleTestResult
{
	enum {Inside, Outside} inout;
	int closestEdge;
	int closestVert;
};


template<typename T>
struct CollisionTable
{
	const int maxSize;
	int* total;
	T* indices;

	__device__ void push_back(T idx)
	{
		int i = atomicAdd(total, 1);
		indices[i] = idx;
	}
};

using TriangleTable = CollisionTable<int2>;
using EdgeTable     = CollisionTable<int3>;

__device__ inline Triangle readTriangle(float4* particles, int3 trid)
{
	return {
		f4tof3( particles[2*trid.x] ),
		f4tof3( particles[2*trid.y] ),
		f4tof3( particles[2*trid.z] ) };
}

// p is assumed to be in the a-b-c plane
// a lot more precise method that the one solving a linear system
__device__ inline float3 barycentric(Triangle tr, float3 p)
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


// find coordinates and "time" (0.0 to 1.0) of the segment - moving plane intersection
__device__ inline Intersection intersectSegmentWithTrianglePlane(
		Triangle tr, Triangle trOld,
		Particle p, Particle pOld)
{
	auto coo = p.r;
	auto cooOld = pOld.r;

	// Precompute scaling factor, in order not to normalize every iteration
	auto n = cross(tr.v1-tr.v0, tr.v2-tr.v0);
	float n_1 = rsqrtf(dot(n, n));

	auto F = [=] (float lambda) {
		float l0 = 1.0f - lambda;
		float l    = lambda;

		float3 r0 = l0*trOld.v0 + l*tr.v0;
		float3 r1 = l0*trOld.v1 + l*tr.v1;
		float3 r2 = l0*trOld.v2 + l*tr.v2;

		float3 x = l0*cooOld + l*coo;
		return dot( n_1 * cross(r1-r0, r2-r0), x - r0 );
	};

	float vold = F(0.0f);
	float vnew = F(1.0f);

	// Not crossing
	if ( vold*vnew > 0.0f ) return {make_float3(0), Triangle(), -1.0f};

	// Particle has crossed the triangle plane
	float alpha = solveLinSearch(F, 1e-7f);

	if (alpha < -0.1f)
		printf("Something awful happened with mesh bounce. F0 = %f,  F = %f\n", F(0.0f), F(1.0f));

	// This is now the collision point with the MOVING triangle
	float l0 = 1.0f - alpha;
	float l  = alpha;
	float3 intPoint = l0*cooOld + l*coo;

	Triangle intTriangle;
	intTriangle.v0 = l0*trOld.v0 + l*tr.v0;
	intTriangle.v1 = l0*trOld.v1 + l*tr.v1;
	intTriangle.v2 = l0*trOld.v2 + l*tr.v2;

	return {intPoint, intTriangle, alpha};
}

__device__ inline InsideTriangleTestResult inOutTest(Triangle tr, float3 p)
{
	const float edgeTolerance = 1e-18f;
	const float vertTolerance = 1e-18f;

	auto length2 = [] (float3 x) {
		return dot(x, x);
	};

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

	InsideTriangleTestResult res{InsideTriangleTestResult::Outside, -1, -1};

	if (s0 > -edgeTolerance && s1 > -edgeTolerance && s2 > -edgeTolerance)
		res.inout = InsideTriangleTestResult::Inside;

	if (length2(p-tr.v0) < vertTolerance) res.closestVert = 0;
	if (length2(p-tr.v1) < vertTolerance) res.closestVert = 1;
	if (length2(p-tr.v2) < vertTolerance) res.closestVert = 2;

	if (fabs(s0) < edgeTolerance) res.closestEdge = 0;
	if (fabs(s1) < edgeTolerance) res.closestEdge = 1;
	if (fabs(s2) < edgeTolerance) res.closestEdge = 2;

	return res;
}

__device__  void findBouncesInCell(
		int pstart, int pend, int globTrid,
		Triangle tr, Triangle trOld,
		PVviewWithOldParticles pvView,
		MeshView mesh,
		EdgeTable edgeTable, TriangleTable triangleTable, int* collisionTimes)
{
	for (int pid=pstart; pid<pend; pid++)
	{
		Particle p, pOld;
		p.   readCoordinate(pvView.particles, pid);
		pOld.readCoordinate(pvView.old_particles, pid);

		// .xyz is the coordinate in-plane of the triangle, .w is the collision "time"
		auto intersection = intersectSegmentWithTrianglePlane(tr, trOld, p, pOld);

		if (intersection.alpha < -0.1f) continue;

		InsideTriangleTestResult result = inOutTest(intersection.tr, intersection.p);

		if (result.inout == InsideTriangleTestResult::Outside) continue;

		atomicMax(collisionTimes+pid, __float_as_int(1.0f - intersection.alpha));

//		if      (result.closestVert >= 0) vertexTable.  push_back({pid, result.closestVert});
//		else
			if  (result.closestEdge >= 0) edgeTable.    push_back({pid, mesh.adjacentTriangles[globTrid*3 + result.closestEdge]});
		else                              triangleTable.push_back({pid, globTrid});
	}
}

__launch_bounds__(128, 6)
static __global__ void findBouncesInMesh(
		OVviewWithNewOldVertices objView,
		PVviewWithOldParticles pvView,
		MeshView mesh,
		CellListInfo cinfo,
		EdgeTable edgeTable, TriangleTable triangleTable, int* collisionTimes)
{
	// About maximum distance a particle can cover in one step
	const float tol = 0.2f;

	// One THREAD per triangle
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int objId = gid / mesh.ntriangles;
	const int trid  = gid % mesh.ntriangles;
	if (objId >= objView.nObjects) return;

	const int3 triangle = mesh.triangles[trid];
	Triangle tr =    readTriangle(objView.vertices     + 2 * mesh.nvertices*objId, triangle);
	Triangle trOld = readTriangle(objView.old_vertices + 2 * mesh.nvertices*objId, triangle);

	// Use old triangle because cell-list is not yet rebuilt now
	const float3 lo = fminf(trOld.v0, fminf(trOld.v1, trOld.v2));
	const float3 hi = fmaxf(trOld.v0, fmaxf(trOld.v1, trOld.v2));

	const int3 cidLow  = cinfo.getCellIdAlongAxes(lo - tol);
	const int3 cidHigh = cinfo.getCellIdAlongAxes(hi + tol);

	int3 cid3;
	for (cid3.z = cidLow.z; cid3.z <= cidHigh.z; cid3.z++)
		for (cid3.y = cidLow.y; cid3.y <= cidHigh.y; cid3.y++)
			{
				cid3.x = cidLow.x;
				int cidLo = max(cinfo.encode(cid3), 0);

				cid3.x = cidHigh.x;
				int cidHi = min(cinfo.encode(cid3)+1, cinfo.totcells);

				int pstart = cinfo.cellStarts[cidLo];
				int pend   = cinfo.cellStarts[cidHi];

				findBouncesInCell(pstart, pend, gid, tr, trOld, pvView, mesh, edgeTable, triangleTable, collisionTimes);
			}
}



/**
 * Reflect the velocity, in the triangle's reference frame
 */
__device__ inline float3 reflectVelocity(float3 n, float kbT, float mass, float seed1, float seed2)
{
	const int maxTries = 50;
	// bounce-back reflection
	// return -initialVelocity;

	// reflection with random scattering
	// according to Maxwell distr
	float2 rand1 = Saru::normal2(seed1, threadIdx.x, blockIdx.x);
	float2 rand2 = Saru::normal2(seed2, threadIdx.x, blockIdx.x);

	float3 r = make_float3(rand1.x, rand1.y, rand2.x);
	for (int i=0; i<maxTries; i++)
	{
		if (dot(r, n) > 0) break;

		float2 rand1 = Saru::normal2(rand2.y, threadIdx.x, blockIdx.x);
		float2 rand2 = Saru::normal2(rand1.y, threadIdx.x, blockIdx.x);
		r = make_float3(rand1.x, rand1.y, rand2.x);
	}
	r = normalize(r) * sqrtf(kbT / mass);

	return r;
}


// Particle with mass M and velocity U0 hits triangle tr (v0, v1, v2)
// into point O. Its new velocity is Unew.
// Vertex masses are m. Treated as rigid and stationary,
// what are the vertex forces induced by the collision?
__device__ inline void triangleForces(
		Triangle tr, float m,
		float3 O_barycentric, float3 U0, float3 Unew, float M,
		float dt,
		float3& f0, float3& f1, float3& f2)
{
	const float tol = 1e-5;

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


static __global__ void performBouncingTriangle(
		OVviewWithNewOldVertices objView,
		PVviewWithOldParticles pvView,
		MeshView mesh,
		int nCollisions, int2* collisionTable, int* collisionTimes,
		const float dt,
		float kbT, float seed1, float seed2)
{
	const float eps = 2e-5f;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= nCollisions) return;

	const int2 pid_trid = collisionTable[gid];
	int pid = pid_trid.x;

	Particle p   (pvView.particles,     pid);
	Particle pOld(pvView.old_particles, pid);
	Particle corrP = p;

	float3 f0 = make_float3(0.0f);
	float3 f1 = make_float3(0.0f);
	float3 f2 = make_float3(0.0f);

	const int trid  = pid_trid.y % mesh.ntriangles;
	const int objId = pid_trid.y / mesh.ntriangles;

	const int3 triangle = mesh.triangles[trid];
	Triangle tr =    readTriangle(objView.vertices     + 2 * mesh.nvertices*objId, triangle);
	Triangle trOld = readTriangle(objView.old_vertices + 2 * mesh.nvertices*objId, triangle);

	auto intersection = intersectSegmentWithTrianglePlane(tr, trOld, p, pOld);

	int minTime = collisionTimes[pid];
	if (1.0f - intersection.alpha != __int_as_float(minTime)) return;

	float3 barycentricCoo = barycentric(intersection.tr, intersection.p);

	const float dt_1 = 1.0f / dt;
	Triangle trVel = { (tr.v0-trOld.v0)*dt_1, (tr.v1-trOld.v1)*dt_1, (tr.v2-trOld.v2)*dt_1 };

	// Position is based on INTERMEDIATE barycentric collision coordinates and FINAL triangle
	const float3 vtri = barycentricCoo.x*trVel.v0 + barycentricCoo.y*trVel.v1 + barycentricCoo.z*trVel.v2;
	const float3 coo  = barycentricCoo.x*tr.v0    + barycentricCoo.y*tr.v1    + barycentricCoo.z*tr.v2;

	const float3 n = normalize(cross(tr.v1-tr.v0, tr.v2-tr.v0));
	float oldSign = dot( pOld.r-trOld.v0, cross(trOld.v1-trOld.v0, trOld.v2-trOld.v0) );

	// new velocity relative to the triangle speed
	float3 newV = reflectVelocity( (oldSign > 0) ? n : -n, kbT, pvView.mass, seed1, seed2 );

	triangleForces(tr, objView.mass, barycentricCoo, p.u - vtri, newV, pvView.mass, dt, f0, f1, f2);

	corrP.r = coo + eps * ((oldSign > 0) ? n : -n);
	corrP.u = newV + vtri;

//	printf("n: %f %f %f, v : %f %f %f, vtri: %f %f %f, cos: %f\n",
//			n.x, n.y, n.z, newV.x, newV.y, newV.z, vtri.x, vtri.y, vtri.z, dot(n, newV));

	corrP.write2Float4(pvView.particles, pid);

//	//if (dot(p.u, p.u) > 20)
//	{
//		printf("Collision %d: particle %d [%f %f %f], [%f %f %f] (old [%f %f %f], [%f %f %f])  to [%f %f %f], [%f %f %f];\n",
//				gid, p.i1, p.r.x, p.r.y, p.r.z,  p.u.x, p.u.y, p.u.z,
//				pOld.r.x, pOld.r.y, pOld.r.z,  pOld.u.x, pOld.u.y, pOld.u.z,
//				corrP.r.x, corrP.r.y, corrP.r.z,  corrP.u.x, corrP.u.y, corrP.u.z);
//		printf("  %d  triangles  %d [%f %f %f] (%f %f %f),  %d [%f %f %f] (%f %f %f),  %d [%f %f %f] (%f %f %f)\n",
//				gid,
//				triangle.x, objView.vertices[2*triangle.x].x, objView.vertices[2*triangle.x].y, objView.vertices[2*triangle.x].z,
//				objView.old_vertices[2*triangle.x].x, objView.old_vertices[2*triangle.x].y, objView.old_vertices[2*triangle.x].z,
//
//				triangle.y, objView.vertices[2*triangle.y].x, objView.vertices[2*triangle.y].y, objView.vertices[2*triangle.y].z,
//				objView.old_vertices[2*triangle.y].x, objView.old_vertices[2*triangle.y].y, objView.old_vertices[2*triangle.y].z,
//
//				triangle.z, objView.vertices[2*triangle.z].x, objView.vertices[2*triangle.z].y, objView.vertices[2*triangle.z].z,
//				objView.old_vertices[2*triangle.z].x, objView.old_vertices[2*triangle.z].y, objView.old_vertices[2*triangle.z].z );
//	}

	atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.x, f0);
	atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.y, f1);
	atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.z, f2);
}

static __global__ void performBouncingEdge(
		OVviewWithNewOldVertices objView,
		PVviewWithOldParticles pvView,
		MeshView mesh,
		int nCollisions, int3* collisionTable, int* collisionTimes,
		const float dt,
		float kbT, float seed1, float seed2)
{
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= nCollisions) return;

	const int3 pid_tr1_tr2 = collisionTable[gid];
	const int pid = pid_tr1_tr2.x;
	const int trid1 = pid_tr1_tr2.y;
	const int trid2 = pid_tr1_tr2.z;

	Particle p   (pvView.particles,     pid);
	p.r = 5.1f * normalize(p.r);
//	Particle pOld(pvView.old_particles, pid);
//	Particle corrP = p;
//
//	float3 f0 = make_float3(0.0f);
//	float3 f1 = make_float3(0.0f);
//	float3 f2 = make_float3(0.0f);
//
//	const int trid  = pid_trid.y % mesh.ntriangles;
//	const int objId = pid_trid.y / mesh.ntriangles;
//
//	const int3 triangle = mesh.triangles[trid];
//	Triangle tr =    readTriangle(objView.vertices     + 2 * mesh.nvertices*objId, triangle);
//	Triangle trOld = readTriangle(objView.old_vertices + 2 * mesh.nvertices*objId, triangle);
//
//	float oldSign = dot( trOld.v0-pOld.r, cross(trOld.v0-trOld.v1, trOld.v0-trOld.v2) );
//	auto intersection = intersectSegmentWithTrianglePlane(tr, trOld, p, pOld);
//
//	float3 barycentricCoo = barycentric(intersection.tr, intersection.p);
//
//	const float dt_1 = 1.0f / dt;
//	Triangle trVel = { (tr.v0-trOld.v0)*dt_1, (tr.v1-trOld.v1)*dt_1, (tr.v2-trOld.v2)*dt_1 };
//
//	// Position is based on INTERMEDIATE barycentric collision coordinates and FINAL triangle
//	const float3 vtri = barycentricCoo.x*trVel.v0 + barycentricCoo.y*trVel.v1 + barycentricCoo.z*trVel.v2;
//	const float3 coo  = barycentricCoo.x*tr.v0    + barycentricCoo.y*tr.v1    + barycentricCoo.z*tr.v2;
//
//	const float3 n = normalize(cross(tr.v1-tr.v0, tr.v2-tr.v0));
//
//	// new velocity relative to the triangle speed
//	float3 newV = reflectVelocity(n, kbT, pvView.mass, seed1, seed2);
//
//	triangleForces(tr, objView.mass, barycentricCoo, p.u - vtri, newV, pvView.mass, dt, f0, f1, f2);
//
//	corrP.r = coo + eps * ((oldSign > 0) ? n : -n);
//	corrP.u = newV + vtri;

	p.write2Float4(pvView.particles, pid);

//	atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.x, f0);
//	atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.y, f1);
//	atomicAdd(objView.vertexForces + mesh.nvertices*objId + triangle.z, f2);
}

