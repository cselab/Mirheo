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

// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
__device__ __forceinline__ float3 computeBarycentric(float3 a, float3 b, float3 c, float3 p)
{
	float3 v0 = b - a, v1 = c - a, v2 = p - a;

	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);
	double invDenom = 1.0 / (double)(d00 * d11 - d01 * d01);

	float l1 = (d11 * d20 - d01 * d21) * invDenom;
	float l2 = (d00 * d21 - d01 * d20) * invDenom;
	float l0 = 1.0f - l1 - l2;

	return make_float3(l0, l1, l2);
}

// Particle with mass M and velocity U0 hits triangle tr (v0, v1, v2)
// into point O. Its new velocity is Unew.
// Vertex masses are m. Treated as rigid and stationary,
// what are the vertex forces induced by the collision?
__device__ __forceinline__ void triangleForces(
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

__device__ __forceinline__ Triangle readTriangle(float4* particles, int3 trid)
{
	return {
		f4tof3( particles[2*trid.x] ),
		f4tof3( particles[2*trid.y] ),
		f4tof3( particles[2*trid.z] ) };
}

// find barycentric coordinates and time (0.0 to 1.0) of the collision
// if at least one of the value returned negative there was no collision
__device__ __forceinline__ float4 intersectParticleTriangleBarycentric(
		Triangle tr, Triangle trOld,
		Particle p, Particle pOld,
		float& oldSign)
{
	auto coo = p.r;
	auto cooOld = pOld.r;

	// Precompute scaling factor, in order not to normalize every iteration
	auto n = cross(tr.v1-tr.v0, tr.v2-tr.v0);
	float n_1 = rsqrtf(dot(n, n));

	auto F = [=] (float lambda) {
		float lOld = 1.0f - lambda;
		float l    = lambda;

		float3 r0 = lOld*trOld.v0 + l*tr.v0;
		float3 r1 = lOld*trOld.v1 + l*tr.v1;
		float3 r2 = lOld*trOld.v2 + l*tr.v2;

		float3 x = lOld*cooOld + l*coo;
		return dot( n_1 * cross(r1-r0, r2-r0), x - r0 );
	};

	float vold = F(0.0f);
	float vnew = F(1.0f);
	oldSign = vold;

	// Particle is not crossing
	// No need for barycentric coordinates, skip it
	if ( vold*vnew > 0.0f ) return make_float4(-1000.0f, -1000.0f, -1000.0f, 1000.0f);

	// Particle has crossed the triangle plane
	float alpha = solveLinSearch(F);

	if (alpha < -0.1f)
		printf("Something awful happened with mesh bounce. F0 = %f,  F = %f\n", F(0.0f), F(1.0f));

	// This is now the collision point with the MOVING triangle
	float lOld = 1.0f - alpha;
	float l    = alpha;
	float3 projected = lOld*cooOld + l*coo;

	// The triangle with which the collision was detected has a timestamp (1.0f - alpha)*dt
	float3 r0 = lOld*trOld.v0 + l*tr.v0;
	float3 r1 = lOld*trOld.v1 + l*tr.v1;
	float3 r2 = lOld*trOld.v2 + l*tr.v2;

	float3 barycentric = computeBarycentric(r0, r1, r2, projected);

	// Return the barycentric coordinates of the projected position
	// on the corresponding moving triangle
	return make_float4(barycentric, alpha);
}


__device__  void findBouncesInCell(
		int pstart, int pend, int globTrid,
		Triangle tr, Triangle trOld,
		PVviewWithOldParticles pvView,
		int* nCollisions, int2* collisionTable, int maxCollisions)
{
	const float tol = 1e-7f;

	for (int pid=pstart; pid<pend; pid++)
	{
		Particle p, pOld;
		p.   readCoordinate(pvView.particles, pid);
		pOld.readCoordinate(pvView.old_particles, pid);

		float oldSign;
		float4 res = intersectParticleTriangleBarycentric(tr, trOld, p, pOld, oldSign);
		float3 barycentricCoo = make_float3(res.x, res.y, res.z);

		if (barycentricCoo.x > -tol && barycentricCoo.y > -tol && barycentricCoo.z > -tol)
		{
			int id = atomicAggInc(nCollisions);
			if (id < maxCollisions)
				collisionTable[id] = make_int2(pid, globTrid);

//			if (id > 500)
//				printf("%d with %d:  %f %f %f\n", p.i1, globTrid,
//						barycentricCoo.x, barycentricCoo.y, barycentricCoo.z);
		}
	}
}

__launch_bounds__(128, 6)
__global__ void findBouncesInMesh(
		OVviewWithOldPartilces objView,
		PVviewWithOldParticles pvView,
		MeshView mesh,
		CellListInfo cinfo,
		int* nCollisions, int2* collisionTable, int maxCollisions)
{
	// About maximum distance a particle can cover in one step
	const float tol = 0.2f;

	// One THREAD per triangle
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int objId = gid / mesh.ntriangles;
	const int trid  = gid % mesh.ntriangles;
	if (objId >= objView.nObjects) return;

	const int3 triangle = mesh.triangles[trid];
	Triangle tr =    readTriangle(objView.particles +     2 * mesh.nvertices*objId, triangle);
	Triangle trOld = readTriangle(objView.old_particles + 2 * mesh.nvertices*objId, triangle);

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

				findBouncesInCell(pstart, pend, gid, tr, trOld, pvView, nCollisions, collisionTable, maxCollisions);
			}
}



__device__ __forceinline__ float2 normal_BoxMuller(float seed)
{
	float u1 = Saru::uniform01(seed, threadIdx.x, blockIdx.x);
	float u2 = Saru::uniform01(u1,   blockIdx.x, threadIdx.x);

	float r = sqrtf(-2.0f * logf(u1));
	float theta = 2.0f * M_PI * u2;

	float2 res;
	sincosf(theta, &res.x, &res.y);
	res *= r;

	return res;
}

/**
 * Reflect the velocity, in the triangle's referece frame
 */
__device__ __forceinline__ float3 reflectVelocity(float3 n, float kbT, float mass, float seed1, float seed2)
{
	const int maxTries = 50;
	// bounce-back reflection
	// return -initialVelocity;

	// reflection with random scattering
	// according to Maxwell distr
	float2 rand1 = normal_BoxMuller(seed1);
	float2 rand2 = normal_BoxMuller(seed2);

	float3 r = make_float3(rand1.x, rand1.y, rand2.x);
	for (int i=0; i<maxTries; i++)
	{
		if (dot(r, n) > 0) break;

		rand1 = normal_BoxMuller(rand2.y);
		rand2 = normal_BoxMuller(rand1.y);
		r = make_float3(rand1.x, rand1.y, rand2.x);
	}
	r = normalize(r) * sqrtf(kbT / mass);

	return r;
}

__global__ void performBouncing(
		OVviewWithOldPartilces objView,
		PVviewWithOldParticles pvView,
		MeshView mesh,
		int nCollisions, int2* collisionTable,
		const float dt,
		float kbT, float seed1, float seed2)
{
	const float eps = 2e-5f;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= nCollisions) return;

	// Collision table has to be sorted by .x - particle ID
	// Since double collisions are veeery rare, we don't need to do smart things
	// Each thread checks around. If all the PIDs around are different - proceed normally
	// If previous PID is the same, just quit
	// If prev is different, but next is the same, process all the following with the same PID

	const int2 pid_trid = collisionTable[gid];
	const int2 pid_trid_prev = gid > 0 ? collisionTable[gid-1] : make_int2(-4242);

	if (pid_trid.x == pid_trid_prev.x) return;

	Particle p   (pvView.particles,     pid_trid.x);
	Particle pOld(pvView.old_particles, pid_trid.x);
	Particle corrP = p;

	float3 f0 = make_float3(0.0f);
	float3 f1 = make_float3(0.0f);
	float3 f2 = make_float3(0.0f);

	float alpha = 1000.0f;
	int firstTriId;

	for (int id = gid; id<nCollisions; id++)
	{
		const int2 nextPid_trid = collisionTable[id];
		if (nextPid_trid.x != pid_trid.x) break;

		const int trid  = nextPid_trid.y % mesh.ntriangles;
		const int objId = nextPid_trid.y / mesh.ntriangles;

		const int3 triangle = mesh.triangles[trid];
		Triangle tr =    readTriangle(objView.particles +     2 * mesh.nvertices*objId, triangle);
		Triangle trOld = readTriangle(objView.old_particles + 2 * mesh.nvertices*objId, triangle);

		float oldSign;
		float4 res = intersectParticleTriangleBarycentric(tr, trOld, p, pOld, oldSign);

		if (res.w < alpha)
		{
			alpha = res.w;
			firstTriId = nextPid_trid.y;
			float3 barycentricCoo = make_float3(res.x, res.y, res.z);

			const float dt_1 = 1.0f / dt;
			Triangle trVel = { (tr.v0-trOld.v0)*dt_1, (tr.v1-trOld.v1)*dt_1, (tr.v2-trOld.v2)*dt_1 };

			const float3 vtri = barycentricCoo.x*trVel.v0 + barycentricCoo.y*trVel.v1 + barycentricCoo.z*trVel.v2;
			const float3 coo  = barycentricCoo.x*tr.v0    + barycentricCoo.y*tr.v1    + barycentricCoo.z*tr.v2;

			const float3 n = normalize(cross(tr.v1-tr.v0, tr.v2-tr.v0));

			// new velocity relative to the triangle speed
			float3 newV = reflectVelocity(n, kbT, pvView.mass, seed1, seed2);

			triangleForces(tr, objView.mass, barycentricCoo, p.u - vtri, newV, pvView.mass, dt, f0, f1, f2);

			corrP.r = coo + eps * ((oldSign > 0) ? n : -n);
			corrP.u = newV + vtri;
		}
	}

	corrP.write2Float4(pvView.particles, pid_trid.x);

	const int trid  = firstTriId % mesh.ntriangles;
	const int objId = firstTriId / mesh.ntriangles;
	const int3 triangle = mesh.triangles[trid];

//	//if (dot(p.u, p.u) > 20)
//	{
//		printf("Collision %d: particle %d [%f %f %f], [%f %f %f] (old [%f %f %f], [%f %f %f])  to [%f %f %f], [%f %f %f];\n",
//				gid, p.i1, p.r.x, p.r.y, p.r.z,  p.u.x, p.u.y, p.u.z,
//				pOld.r.x, pOld.r.y, pOld.r.z,  pOld.u.x, pOld.u.y, pOld.u.z,
//				corrP.r.x, corrP.r.y, corrP.r.z,  corrP.u.x, corrP.u.y, corrP.u.z);
//		printf("  %d  triangles  %d [%f %f %f] (%f %f %f),  %d [%f %f %f] (%f %f %f),  %d [%f %f %f] (%f %f %f)\n",
//				gid,
//				triangle.x, objView.particles[2*triangle.x].x, objView.particles[2*triangle.x].y, objView.particles[2*triangle.x].z,
//				objView.old_particles[2*triangle.x].x, objView.old_particles[2*triangle.x].y, objView.old_particles[2*triangle.x].z,
//
//				triangle.y, objView.particles[2*triangle.y].x, objView.particles[2*triangle.y].y, objView.particles[2*triangle.y].z,
//				objView.old_particles[2*triangle.y].x, objView.old_particles[2*triangle.y].y, objView.old_particles[2*triangle.y].z,
//
//				triangle.z, objView.particles[2*triangle.z].x, objView.particles[2*triangle.z].y, objView.particles[2*triangle.z].z,
//				objView.old_particles[2*triangle.z].x, objView.old_particles[2*triangle.z].y, objView.old_particles[2*triangle.z].z );
//	}
//
//
//	//if (length(f0) > 5000)
//		printf("%d force %f %f %f\n", mesh.nvertices*objId + triangle.x, f0.x, f0.y, f0.z);
//
//	//if (length(f1) > 5000)
//		printf("%d force %f %f %f\n", mesh.nvertices*objId + triangle.y, f1.x, f1.y, f1.z);
//
//	//if (length(f2) > 5000)
//		printf("%d force %f %f %f\n", mesh.nvertices*objId + triangle.z, f2.x, f2.y, f2.z);

	atomicAdd(objView.forces + mesh.nvertices*objId + triangle.x, f0);
	atomicAdd(objView.forces + mesh.nvertices*objId + triangle.y, f1);
	atomicAdd(objView.forces + mesh.nvertices*objId + triangle.z, f2);
}


