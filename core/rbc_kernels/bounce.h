#pragma once

#include <core/pvs/rbc_vector.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/bounce_solver.h>

// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
__device__ __forceinline__ float3 computeBarycentric(float3 a, float3 b, float3 c, float3 p)
{
	float3 v0 = b - a, v1 = c - a, v2 = p - a;

	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);
	float invDenom = 1.0f / (d00 * d11 - d01 * d01);

	float l1 = (d11 * d20 - d01 * d21) * invDenom;
	float l2 = (d00 * d21 - d01 * d20) * invDenom;
	float l0 = 1.0f - l1 - l2;

	return make_float3(l0, l1, l2);
}

// Particle with mass M and velocity U hits triangle (x0, x1, x2)
// into point O. Vertex masses are m. Treated as rigid and stationary,
// what are the vertex forces induced by the collision?
__device__ __forceinline__ void triangleForces(
		float3 x0, float3 x1, float3 x2, float m,
		float3 O_barycentric, float3 U, float M,
		float dt,
		float3& f0, float3& f1, float3& f2)
{
	auto len2 = [] (float3 x) {
		return dot(x, x);
	};

	const float3 n = normalize(cross(x1-x0, x2-x0));

	const float IU_ortI = dot(U, n);
	const float3 U_par = U - IU_ortI * n;

	const float a = 2.0f*M/m * IU_ortI;
	const float v0_ort = O_barycentric.x * a;
	const float v1_ort = O_barycentric.y * a;
	const float v2_ort = O_barycentric.z * a;

	const float3 C = 0.333333333f * (x0+x1+x2);
	const float3 Vc = 0.666666666f * M/m * U_par;

	const float3 O = O_barycentric.x * x0 + O_barycentric.y * x1 + O_barycentric.z * x2;
	const float3 L = 2.0f*M * cross(C-O, U_par);

	const float J = m * (len2(C-x0) + len2(C-x1) + len2(C-x2));
	const float w = -dot(L, n) / J;

	const float3 orth_r0 = cross(C-x0, n);
	const float3 orth_r1 = cross(C-x1, n);
	const float3 orth_r2 = cross(C-x2, n);

	const float3 u0 = w * orth_r0;
	const float3 u1 = w * orth_r1;
	const float3 u2 = w * orth_r2;

	const float3 v0 = v0_ort*n + Vc + u0;
	const float3 v1 = v1_ort*n + Vc + u1;
	const float3 v2 = v2_ort*n + Vc + u2;

	const float invdt = 1.0f / dt;
	f0 = v0 * m * invdt;
	f1 = v1 * m * invdt;
	f2 = v2 * m * invdt;
}

struct Triangle
{
	float3 v0, v1, v2;
};

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

	auto F = [=] (float lambda) {
		float lOld = 1.0f - lambda;
		float l    = lambda;

		float3 r0 = lOld*trOld.v0 + l*tr.v0;
		float3 r1 = lOld*trOld.v1 + l*tr.v1;
		float3 r2 = lOld*trOld.v2 + l*tr.v2;

		float3 x = lOld*cooOld + l*coo;
		return dot( normalize(cross(r1-r0, r2-r0)), x - r0 );
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


__device__ __forceinline__ void findBouncesInCell(
		int pstart, int pend, int globTrid,
		Triangle tr, Triangle trOld,
		PVview_withOldParticles pvView, int* nCollisions, int2* collisionTable)
{
	const float tol = 1e-6f;

	for (int pid=pstart; pid<pend; pid++)
	{
		Particle p(pvView.particles, pid);
		Particle pOld(pvView.old_particles, pid);

		float oldSign;
		float4 res = intersectParticleTriangleBarycentric(tr, trOld, p, pOld, oldSign);
		float3 barycentricCoo = make_float3(res.x, res.y, res.z);

		if (barycentricCoo.x > -tol && barycentricCoo.y > -tol && barycentricCoo.z > -tol)
		{
			int id = atomicAggInc(nCollisions);
			collisionTable[id] = make_int2(pid, globTrid);
		}
	}
}


__device__ inline bool isCellCrossingTriangle(float3 cornerCoo, float3 len, float3 n, float3 r0, float tol)
{
	int pos = 0, neg = 0;

#pragma unroll
	for (int i=0; i<2; i++)
#pragma unroll
		for (int j=0; j<2; j++)
#pragma unroll
			for (int k=0; k<2; k++)
			{
				// Value in the cell corner
				const float3 shift = make_float3(i ? len.x : 0.0f, j ? len.y : 0.0f, k ? len.z : 0.0f);
				const float s = dot(n, cornerCoo + shift - r0);
				if (s >  tol) pos++;
				if (s < -tol) neg++;
			}

	return (pos != 8 && neg != 8);
}

__global__ void findBouncesInMesh(
		OVviewWithOldPartilces objView,
		PVview_withOldParticles pvView,
		MeshView mesh,
		CellListInfo cinfo,
		int* nCollisions, int2* collisionTable)
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

	float3 f0 = make_float3(0.0f);
	float3 f1 = make_float3(0.0f);
	float3 f2 = make_float3(0.0f);

	int3 cid3;
	for (cid3.x = cidLow.x; cid3.x <= cidHigh.x; cid3.x++)
		for (cid3.y = cidLow.y; cid3.y <= cidHigh.y; cid3.y++)
			for (cid3.z = cidLow.z; cid3.z <= cidHigh.z; cid3.z++)
			{
				const float3 v000 = make_float3(cid3) * cinfo.h - cinfo.localDomainSize*0.5f;
				int cid = cinfo.encode(cid3);
				if (cid < 0 || cid >= cinfo.totcells) continue;

				const bool valid = isCellCrossingTriangle(
						v000, cinfo.h, cross(trOld.v1-trOld.v0, trOld.v2-trOld.v0), trOld.v0, tol );

				if (valid)
				{
					int pstart = cinfo.cellStarts[cid];
					int pend = cinfo.cellStarts[cid+1];

					findBouncesInCell(pstart, pend, gid, tr, trOld,
							pvView, nCollisions, collisionTable);
				}
			}
}


__global__ void performBouncing(
		OVviewWithOldPartilces objView,
		PVview_withOldParticles pvView,
		MeshView mesh,
		int nCollisions, int2* collisionTable,
		const float dt)
{
	const float eps = 2e-6f;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= nCollisions) return;

	// Collision table has to be sorted by .x - particle ID
	// Since double collisions are veeery rare, we don't need to do smart things
	// Each thread checks around. If all the PIDs around are different - proceed normally
	// If previous PID is the same, just quit
	// If prev is different, but next is the same, process all the following with the same PID

	const int2 pid_trid = collisionTable[gid];
	const int2 pid_trid_prev = gid > 0 ? collisionTable[gid-1] : make_int2(-1);

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

			triangleForces(tr.v0, tr.v1, tr.v2, objView.mass, barycentricCoo, p.u - vtri, pvView.mass, dt, f0, f1, f2);

			float3 newV = 2.0f*vtri-p.u;

			const float3 n = normalize(cross(tr.v1-tr.v0, tr.v2-tr.v0));
			corrP.r = coo + eps * n * ((oldSign > 0) ? 5.0f : -5.0f);

			corrP.u = newV;
		}
	}

	corrP.write2Float4(pvView.particles, pid_trid.x);

	const int trid  = firstTriId % mesh.ntriangles;
	const int objId = firstTriId / mesh.ntriangles;
	const int3 triangle = mesh.triangles[trid];

	atomicAdd(objView.forces + mesh.nvertices*objId + triangle.x, f0);
	atomicAdd(objView.forces + mesh.nvertices*objId + triangle.y, f1);
	atomicAdd(objView.forces + mesh.nvertices*objId + triangle.z, f2);

}



