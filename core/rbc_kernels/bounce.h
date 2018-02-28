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

__device__ inline bool isInside(Triangle tr, float3 p)
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


	return (s0 > -edgeTolerance && s1 > -edgeTolerance && s2 > -edgeTolerance);
}


__device__ inline void sortUnique3(float2* v)
{
	const float tol = 1e-4f;

	auto swap = [] (float2& a, float2& b) {
		float2 tmp = a;
		a = b;
		b = tmp;
	};

	if (fabsf(v[0].x - v[1].x) < tol)
		v[1] = make_float2(2.0f, -1.0f);

	if (fabsf(v[0].x - v[2].x) < tol)
		v[2] = make_float2(2.0f, -1.0f);

	if (fabsf(v[1].x - v[2].x) < tol)
		v[2] = make_float2(2.0f, -1.0f);


	if (v[0].x > v[1].x) swap(v[0], v[1]);
	if (v[0].x > v[2].x) swap(v[0], v[2]);
	if (v[1].x > v[2].x) swap(v[1], v[2]);
}

// find "time" (0.0 to 1.0) of the segment - moving triangle intersection
// returns -1 is no intersection
// sets intPoint and intTriangle if intersection found
__device__ static float intersectSegmentWithTriangle(
		Triangle tr, Triangle trOld,
		Particle p, Particle pOld,

		float3& intPoint,
		Triangle& intTriangle,
		float& intSign,
		int trid = -1)
{
	const float tol = 2e-6f;

	const float3 x  = pOld.r;
	const float3 v0 = trOld.v0;
	const float3 v1 = trOld.v1;
	const float3 v2 = trOld.v2;

	const float3 dx  = p.r - x;
	const float3 dv0 = tr.v0 - v0;
	const float3 dv1 = tr.v1 - v1;
	const float3 dv2 = tr.v2 - v2;

	// n(t) = na*t^2 + nb*t + nc
	const float3 na = cross(dv1-dv0, dv2-dv0);
	const float3 nb = cross(v1-v0, dv2-dv0) + cross(dv1-dv0, dv2-dv0);
	const float3 nc = cross(v1-v0, v2-v0);

	const float3 dx_dv0 = dx-dv0;
	const float3 x_v0 = x-v0;

	const float a =                 dot(dx_dv0, na);
	const float b = dot(x_v0, na) + dot(dx_dv0, nb);
	const float c = dot(x_v0, nb) + dot(dx_dv0, nc);
	const float d = dot(x_v0, nc);

	// precompute scaling factor
	auto n = cross(tr.v1-tr.v0, tr.v2-tr.v0);
	float n_1 = rsqrtf(dot(n, n));

	// Distance to a triangle
	auto F = [=] (float t) {
		float3 v0t = v0 + t*dv0;
		float3 v1t = v1 + t*dv1;
		float3 v2t = v2 + t*dv2;

		float3 xt = x + t*dx;
		return  n_1 * dot( xt - v0t, cross(v1t-v0t, v2t-v0t) );
	};

	// d / dt (Distance)
	auto F_prime = [=] (float t) {
		float3 v0t = v0 + t*dv0;
		float3 v1t = v1 + t*dv1;
		float3 v2t = v2 + t*dv2;

		float3 nt = cross(v1t-v0t, v2t-v0t);

		float3 xt = x + t*dx;
		return  n_1 * ( dot(dx-dv0, nt) + dot(xt-v0t, cross(dv1-dv0, v2t-v0t) + cross(v1t-v0t, dv2-dv0)) );
	};

	// Has side-effects!!
	intSign = -1.0f;
	auto checkIfInside = [&] (float alpha) {
		intPoint = x + alpha*dx;

		intTriangle.v0 = v0 + alpha*dv0;
		intTriangle.v1 = v1 + alpha*dv1;
		intTriangle.v2 = v2 + alpha*dv2;

		intSign = -F_prime(alpha);

		return isInside(intTriangle, intPoint);
	};

	float2 roots[3];
	roots[0] = solveNewton(F, F_prime, 0.0f);
	roots[2] = solveNewton(F, F_prime, 1.0f);

	float change = F(0.0f) * F(1.0f);

	if (change < 0.0f)
	{
		// Three roots
		if (roots[0].x >= 0.0f && roots[0].x <= 1.0f && fabsf(roots[0].y) < 2e-6f &&
			roots[2].x >= 0.0f && roots[2].x <= 1.0f && fabsf(roots[2].y) < 2e-6f)
			roots[1] = solveLinSearch_verbose(F, roots[0].x + 1e-5f/fabsf(F_prime(roots[0].x)), roots[2].x - 1e-5f/fabsf(F_prime(roots[2].x)));
		else // One root
			roots[1] = solveLinSearch_verbose(F, 0.0f, 1.0f);
	}
	else  // Maybe two roots
	{
		float2 newtonRoot;

		if (roots[0].x >= 0.0f && roots[0].x <= 1.0f && fabsf(roots[0].y) < 2e-6f)
			newtonRoot = roots[0];

		if (roots[2].x >= 0.0f && roots[2].x <= 1.0f && fabsf(roots[2].y) < 2e-6f)
			newtonRoot = roots[2];

//		if (p.i1 == 69211  && trid == 384)
//			printf("one root %f,  [%f ** %f : %f : %f ** %f]\n",
//					newtonRoot.x, F(0.0f),
//					F(newtonRoot.x - 1e-5f/fabsf(F_prime(newtonRoot.x))),
//					F(newtonRoot.x),
//					F(newtonRoot.x + 1e-5f/fabsf(F_prime(newtonRoot.x))),
//					F(1.0f));

		roots[1] = solveLinSearch_verbose(F, newtonRoot.x + 1e-5f/fabsf(F_prime(newtonRoot.x)), 1.0f);

		if (fabsf(roots[1].y) > 2e-6f)
		roots[1] = solveLinSearch_verbose(F, 0.0f, newtonRoot.x - 1e-5f/fabsf(F_prime(newtonRoot.x)));
	}


	sortUnique3(roots);


	if (p.i1 == 69211  && ((trid == 341 || trid == 384 || trid == 14 || trid ==  9 || trid == 964)))
		printf("Roots %d:: r1 %f -> %f (%f),  r2 %f -> %f (%f),  r3 %f -> %f (%f)\n", trid,
				roots[0].x, roots[0].y, F_prime(roots[0].x),
				roots[1].x, roots[1].y, F_prime(roots[1].x),
				roots[2].x, roots[2].y, F_prime(roots[2].x));

	if ( fabs(roots[0].y) < 2e-6f && roots[0].x >= 0.0f && roots[0].x <= 1.0f )
		if (checkIfInside(roots[0].x)) return roots[0].x;

	if ( fabs(roots[1].y) < 2e-6f && roots[1].x >= 0.0f && roots[1].x <= 1.0f )
		if (checkIfInside(roots[1].x)) return roots[1].x;

	if ( fabs(roots[2].y) < 2e-6f && roots[2].x >= 0.0f && roots[2].x <= 1.0f )
		if (checkIfInside(roots[2].x)) return roots[2].x;

	return -1.0f;
}


__device__ void findBouncesInCell(
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

		float3 intP;
		Triangle intTr;
		float intSign;
		float alpha = intersectSegmentWithTriangle(tr, trOld, p, pOld,  intP, intTr, intSign, globTrid);

		float oldSign = dot( pOld.r-trOld.v0, normalize(cross(trOld.v1-trOld.v0, trOld.v2-trOld.v0)) );
		float newSign = dot( p.r-tr.v0, normalize(cross(tr.v1-tr.v0, tr.v2-tr.v0)) );

//		if (p.i1 == 69211 && oldSign * newSign < 0.0f)
//		{
//			printf("tr %d ([%f %f %f]  [%f %f %f]  [%f %f %f]), %f -> %f;  alpha %f\n",
//					globTrid,
//					tr.v0.x, tr.v0.y, tr.v0.z,
//					tr.v1.x, tr.v1.y, tr.v1.z,
//					tr.v2.x, tr.v2.y, tr.v2.z,
//					oldSign, newSign, alpha);
//		}

		if ( p.i1 == 69211 && (globTrid == 341 || globTrid == 384 || globTrid == 14 || globTrid ==  9 || globTrid == 964) )
		{
			float3 bc = barycentric(intTr, intP);
			printf("tr %d , old %f -> new %f;  alpha: %f, sign %f, coo: %f %f %f\n  p: [%f, %f, %f], tr [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n",
					globTrid, oldSign, newSign, alpha, intSign,
					bc.x, bc.y, bc.z,
					intP.x, intP.y, intP.z,
					intTr.v0.x, intTr.v0.y, intTr.v0.z,
					intTr.v1.x, intTr.v1.y, intTr.v1.z,
					intTr.v2.x, intTr.v2.y, intTr.v2.z);
		}
//
//		if ( p.i1 == 69211 && (globTrid == 341 ) )
//		{
//			printf("INTERSECTION CHECK tr %d , OLD p: [%f, %f, %f], tr [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n"
//					"                           NEW p: [%f, %f, %f], tr [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n",
//					globTrid,
//					pOld.r.x, pOld.r.y, pOld.r.z,
//					trOld.v0.x, trOld.v0.y, trOld.v0.z,
//					trOld.v1.x, trOld.v1.y, trOld.v1.z,
//					trOld.v2.x, trOld.v2.y, trOld.v2.z,
//
//					p.r.x, p.r.y, p.r.z,
//					tr.v0.x, tr.v0.y, tr.v0.z,
//					tr.v1.x, tr.v1.y, tr.v1.z,
//					tr.v2.x, tr.v2.y, tr.v2.z);
//		}


		if (alpha < -0.1f) continue;

		atomicMax(collisionTimes+pid, __float_as_int(1.0f - alpha));

//		if      (result.closestVert >= 0) vertexTable.  push_back({pid, result.closestVert});
//		else
//			if  (result.closestEdge >= 0) edgeTable.    push_back({pid, mesh.adjacentTriangles[globTrid*3 + result.closestEdge]});
//		else
		triangleTable.push_back({pid, globTrid});
	}
}


// =====================================================
// Small convenience functions
// =====================================================

template<typename T, typename... Args>
__device__ inline T fmin_vec(T v, Args... args)
{
	return fminf(v, fmin_vec(args...));
}

template<typename T>
__device__ inline T fmin_vec(T v)
{
	return v;
}

template<typename T, typename... Args>
__device__ inline T fmax_vec(T v, Args... args)
{
	return fmaxf(v, fmax_vec(args...));
}

template<typename T>
__device__ inline T fmax_vec(T v)
{
	return v;
}

// =====================================================
// =====================================================


__launch_bounds__(128, 6)
static __global__ void findBouncesInMesh(
		OVviewWithNewOldVertices objView,
		PVviewWithOldParticles pvView,
		MeshView mesh,
		CellListInfo cinfo,
		EdgeTable edgeTable, TriangleTable triangleTable, int* collisionTimes)
{
	// About maximum distance a particle can cover in one step
	const float tol = 0.3f;

	// One THREAD per triangle
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int objId = gid / mesh.ntriangles;
	const int trid  = gid % mesh.ntriangles;
	if (objId >= objView.nObjects) return;

	const int3 triangle = mesh.triangles[trid];
	Triangle tr =    readTriangle(objView.vertices     + 2 * mesh.nvertices*objId, triangle);
	Triangle trOld = readTriangle(objView.old_vertices + 2 * mesh.nvertices*objId, triangle);

	const float3 lo = fmin_vec(trOld.v0, trOld.v1, trOld.v2, tr.v0, tr.v1, tr.v2);
	const float3 hi = fmax_vec(trOld.v0, trOld.v1, trOld.v2, tr.v0, tr.v1, tr.v2);

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
__device__ inline void triangleForces(
		Triangle tr, float m,
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


static __global__ void performBouncingTriangle(
		OVviewWithNewOldVertices objView,
		PVviewWithOldParticles pvView,
		MeshView mesh,
		int nCollisions, int2* collisionTable, int* collisionTimes,
		const float dt,
		float kbT, float seed1, float seed2)
{
	const float eps = 5e-5f;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= nCollisions) return;

	const int2 pid_trid = collisionTable[gid];
	int pid = pid_trid.x;

	Particle p   (pvView.particles,     pid);
	Particle pOld(pvView.old_particles, pid);
	Particle corrP = p;

	float3 f0, f1, f2;

	const int trid  = pid_trid.y % mesh.ntriangles;
	const int objId = pid_trid.y / mesh.ntriangles;

	const int3 triangle = mesh.triangles[trid];
	Triangle tr =    readTriangle(objView.vertices     + 2 * mesh.nvertices*objId, triangle);
	Triangle trOld = readTriangle(objView.old_vertices + 2 * mesh.nvertices*objId, triangle);

	float3 intPoint;
	Triangle intTriangle;
	float intSign;
	float alpha = intersectSegmentWithTriangle(tr, trOld, p, pOld, intPoint, intTriangle, intSign);

	int minTime = collisionTimes[pid];

	if (1.0f - alpha != __int_as_float(minTime)) return;

	if (p.i1 == 69211)
		printf("COLLISION w %d !!!\n\n\n", pid_trid.y);

	float3 barycentricCoo = barycentric(intTriangle, intPoint);

	const float dt_1 = 1.0f / dt;
	Triangle trVel = { (tr.v0-trOld.v0)*dt_1, (tr.v1-trOld.v1)*dt_1, (tr.v2-trOld.v2)*dt_1 };

	// Position is based on INTERMEDIATE barycentric collision coordinates and FINAL triangle
	const float3 vtri = barycentricCoo.x*trVel.v0 + barycentricCoo.y*trVel.v1 + barycentricCoo.z*trVel.v2;
	const float3 coo  = barycentricCoo.x*tr.v0    + barycentricCoo.y*tr.v1    + barycentricCoo.z*tr.v2;

	const float3 n = normalize(cross(tr.v1-tr.v0, tr.v2-tr.v0));
	float oldSign = dot( pOld.r-trOld.v0, cross(trOld.v1-trOld.v0, trOld.v2-trOld.v0) );

	// new velocity relative to the triangle speed
	float3 newV = reflectVelocity( (intSign * oldSign > 0) ? n : -n, kbT, pvView.mass, seed1, seed2 );

	triangleForces(tr, objView.mass, barycentricCoo, p.u - vtri, newV, pvView.mass, dt, f0, f1, f2);

	corrP.r = coo + eps * ((intSign * oldSign > 0) ? n : -n);
	corrP.u = newV + vtri;

	float sign = dot( corrP.r-tr.v0, cross(tr.v1-tr.v0, tr.v2-tr.v0) );

//	if (oldSign < 0.0f || sign < 0.0f)
//		printf("%f  vs  %f\n", oldSign, sign);

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
	p.r = make_float3(9) + normalize(p.r);
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

