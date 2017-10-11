#pragma once

#include <core/pvs/rbc_vector.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/bounce_solver.h>


// Expand the triangle by extra for robustness
__device__ __host__ __forceinline__ void expandBy(float3& r0, float3& r1, float3& r2, float extra)
{
	auto invLen = [] (float3 x) {
		return rsqrt(dot(x, x));
	};

	float3 center = 0.33333333333f * (r0 + r1 + r2);
	r0 = (r0 - center) * (1.0f + extra * invLen(r0-center)) + center;
	r1 = (r1 - center) * (1.0f + extra * invLen(r1-center)) + center;
	r2 = (r2 - center) * (1.0f + extra * invLen(r2-center)) + center;
}


// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
__device__ __host__ __forceinline__ float3 computeBarycentric(float3 a, float3 b, float3 c, float3 p)
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
__device__ __host__ __forceinline__ void triangleForces(
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

// find barycentric coordinates and time (0.0 to 1.0) of the collision
// if at least one of the value returned negative there was no collision
__device__ __forceinline__ float4 intersectParticleTriangleBarycentric(
		Particle v0, Particle v1, Particle v2, float3 n,
		Particle p, float dt, float threshold, float& oldSign)
{
	// Increase the triangle a bit for robustness
	expandBy(v0.r, v1.r, v2.r, threshold);

	auto F = [=] (float lambda) {
		float t = (1.0f - lambda)*dt;
		float3 r0 = v0.r - v0.u*t;
		float3 r1 = v1.r - v1.u*t;
		float3 r2 = v2.r - v2.u*t;

		float3 x  = p .r - p .u*t;

		return dot( normalize(cross(r1-r0, r2-r0)), x - r0 );
	};

	float vold = F(0.0f);
	float vnew = F(1.0f);
	oldSign = vold;

//	int trid = (blockIdx.x * blockDim.x + threadIdx.x);
//	if (p.i1 == 20886 && trid == 184)
//		printf(" %d with tr %d: %f  ->  %f\n",
//					p.i1, trid, vold, vnew);

	// Particle is far and not crossing
	// No need for barycentric coordinates, skip it
	if ( fabs(vnew) > threshold && vold*vnew >= 0.0f ) return make_float4(-1000.0f, -1000.0f, -1000.0f, 1000.0f);

	const float3 C = 0.333333333f * (v0.r+v1.r+v2.r);

	float3 projected;
	float alpha = 1.0f;

	// Particle is in a near band but it didn't cross the triangle
	if (vnew*vold >= 0.0f && fabs(vnew) < threshold)
		projected = p.r - dot(p.r-v0.r, n)*n;
	else
	{
		// Particle has crossed the triangle plane
		alpha = solveLinSearch(F);

		// This should NEVER happen
		if (alpha < -0.1f)
			printf("Something awful happened with mesh bounce. F0 = %f,  F = %f\n", F(0.0f), F(1.0f));

		// This is the collision point with the MOVING triangle
		projected = p.r - p.u*(1.0f - alpha)*dt;
	}

	// The triangle with which the collision was detected has a timestamp (1.0f - alpha)*dt
	float t = (1.0f - alpha)*dt;
	float3 r0 = v0.r - v0.u*t;
	float3 r1 = v1.r - v1.u*t;
	float3 r2 = v2.r - v2.u*t;

	float3 barycentric = computeBarycentric(r0, r1, r2, projected);

//	if (p.i1 == 17040 )//&& trid == 153)
//		printf(" %d with tr %d: %f  ->  %f , dist to center: %f. T = %f,  bar_coo = %f %f %f,  real coo  %f %f %f\n",
//				p.i1, (blockIdx.x * blockDim.x + threadIdx.x) / warpSize,
//				vold, vnew, sqrtf(dot(p.r - C, p.r - C)), alpha, barycentric.x, barycentric.y, barycentric.z,
//				p.r.x, p.r.y, p.r.z);

	// Return the barycentric coordinates of the projected position
	// on the corresponding moving triangle
	return make_float4(barycentric, alpha);
}


// FIXME: add different masses
__device__ __forceinline__ void findBouncesInCell(
		int pstart, int pend, int globTrid,
		Particle v0, Particle v1, Particle v2,
		const float4* coosvels, int* nCollisions, int2* collisionTable,
		const float dt)
{
	const float threshold = 2e-6f;

	const float3 n = normalize(cross(v1.r-v0.r, v2.r-v0.r));

	for (int pid=pstart; pid<pend; pid++)
	{
		Particle p(coosvels[2*pid], coosvels[2*pid+1]);

		float oldSign;
		float4 res = intersectParticleTriangleBarycentric(v0, v1, v2, n, p, dt, threshold, oldSign);
		float3 barycentricCoo = make_float3(res.x, res.y, res.z);

		if (barycentricCoo.x >= 0.0f && barycentricCoo.y >= 0.0f && barycentricCoo.z >= 0.0f)
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
		const float4* __restrict__ coosvels, CellListInfo cinfo,
		int* nCollisions, int2* collisionTable,
		const int nObj, const int nvertices, const int ntriangles,
		const int3* __restrict__ triangles, const float4* __restrict__ objCoosvels,
		const float dt)
{
	// About maximum distance a particle can cover in one step
	const float tol = 0.1f;
	const float eps = 2e-6f;

	// One THREAD per triangle
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int objId = gid / ntriangles;
	const int trid  = gid % ntriangles;
	if (objId >= nObj) return;

	const int3 triangle = triangles[trid];
	Particle v0 = Particle(objCoosvels, nvertices*objId + triangle.x);
	Particle v1 = Particle(objCoosvels, nvertices*objId + triangle.y);
	Particle v2 = Particle(objCoosvels, nvertices*objId + triangle.z);
	expandBy(v0.r, v1.r, v2.r, 2.0f*eps);
	float3 n = cross(v1.r-v0.r, v2.r-v0.r);

	const float3 lo = fminf(v0.r, fminf(v1.r, v2.r));
	const float3 hi = fmaxf(v0.r, fmaxf(v1.r, v2.r));

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
				const bool valid = isCellCrossingTriangle(v000, cinfo.h, n, v0.r, tol);

				if (valid)
				{
					int cid = cinfo.encode(cid3);
					int pstart = cinfo.cellStarts[cid];
					int pend = cinfo.cellStarts[cid+1];

					findBouncesInCell(pstart, pend, blockIdx.x * blockDim.x + threadIdx.x, v0, v1, v2,
								coosvels, nCollisions, collisionTable,  dt);
				}
			}

}


__global__ void performBouncing(int nCollisions, int2* collisionTable,
		float4* coosvels, const float partMass,
		const int nvertices, const int ntriangles,
		const int3* __restrict__ triangles, const float4* __restrict__ objCoosvels, float* objForces, float vertMass,
		const float dt)
{
	const float eps = 2e-6f;

	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= nCollisions) return;

	// Collision table has to be sorted by .x - particle ID
	// Since double collisions are veeery rare, we don't need to do smart things
	// Each thread checks around. If all the PIDs around are different - proceed normally
	// If previous PID is the same, just quit
	// If prev is different, but next is the same, process all the next with the same PID

	const int2 pid_trid = collisionTable[gid];
	const int2 pid_trid_prev = gid > 0 ? collisionTable[gid-1] : make_int2(-1);

	//printf("GID  %d,  Collsion: p %d vs tr %d\n", gid, pid_trid.x, pid_trid.y);

	if (pid_trid.x == pid_trid_prev.x) return;

	Particle p(coosvels, pid_trid.x);
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

//		if (id > gid) printf("Found double collision:  p %d tr %d  and  p %d tr %d\n",
//				pid_trid.x, pid_trid.y, nextPid_trid.x, nextPid_trid.y);

		const int trid  = nextPid_trid.y % ntriangles;
		const int objId = nextPid_trid.y / ntriangles;

		const int3 triangle = triangles[trid];
		Particle v0 = Particle(objCoosvels, nvertices*objId + triangle.x);
		Particle v1 = Particle(objCoosvels, nvertices*objId + triangle.y);
		Particle v2 = Particle(objCoosvels, nvertices*objId + triangle.z);

		float oldSign;
		expandBy(v0.r, v1.r, v2.r, 2.0f*eps);
		float3 n = normalize(cross(v1.r-v0.r, v2.r-v0.r));

		float4 res = intersectParticleTriangleBarycentric(v0, v1, v2, n, p, dt, eps, oldSign);

		if (res.w < alpha)
		{
			alpha = res.w;
			firstTriId = nextPid_trid.y;
			float3 barycentricCoo = make_float3(res.x, res.y, res.z);

			const float3 vtri = barycentricCoo.x*v0.u + barycentricCoo.y*v1.u + barycentricCoo.z*v2.u;
			const float3 coo  = barycentricCoo.x*v0.r + barycentricCoo.y*v1.r + barycentricCoo.z*v2.r;

			triangleForces(v0.r, v1.r, v2.r, vertMass, barycentricCoo, p.u - vtri, partMass, dt, f0, f1, f2);

			float3 newV = 2.0f*vtri-p.u;
			corrP.r = coo + eps * n * ((oldSign > 0) ? 5.0f : -5.0f);
			corrP.u = newV;
		}
	}

	coosvels[2*pid_trid.x]   = corrP.r2Float4();
	coosvels[2*pid_trid.x+1] = corrP.u2Float4();

	const int trid  = firstTriId % ntriangles;
	const int objId = firstTriId / ntriangles;
	const int3 triangle = triangles[trid];

	atomicAdd(objForces + 3*(nvertices*objId + triangle.x),   f0.x);
	atomicAdd(objForces + 3*(nvertices*objId + triangle.x)+1, f0.y);
	atomicAdd(objForces + 3*(nvertices*objId + triangle.x)+2, f0.z);

	atomicAdd(objForces + 3*(nvertices*objId + triangle.y),   f1.x);
	atomicAdd(objForces + 3*(nvertices*objId + triangle.y)+1, f1.y);
	atomicAdd(objForces + 3*(nvertices*objId + triangle.y)+2, f1.z);

	atomicAdd(objForces + 3*(nvertices*objId + triangle.z),   f2.x);
	atomicAdd(objForces + 3*(nvertices*objId + triangle.z)+1, f2.y);
	atomicAdd(objForces + 3*(nvertices*objId + triangle.z)+2, f2.z);
}



