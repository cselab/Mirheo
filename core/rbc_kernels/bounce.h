#pragma once

#include <core/rbc_vector.h>
#include <core/cuda_common.h>
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
	f0 += v0 * m * invdt;
	f1 += v1 * m * invdt;
	f2 += v2 * m * invdt;
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

		return dot( normalize(cross(r1-r0, r2-r0)), x - r0);
	};

	float vold = F(0.0f);
	float vnew = F(1.0f);
	oldSign = vold;

	int trid = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
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

//	if (p.i1 == 38669 )//&& trid == 153)
//		printf(" %d with tr %d: %f  ->  %f , dist to center: %f. T = %f,  bar_coo = %f %f %f,  real coo  %f %f %f\n",
//				p.i1, (blockIdx.x * blockDim.x + threadIdx.x) / warpSize,
//				vold, vnew, sqrtf(dot(p.r - C, p.r - C)), alpha, barycentric.x, barycentric.y, barycentric.z,
//				p.r.x, p.r.y, p.r.z);

	// Return the barycentric coordinates of the projected position
	// on the corresponding moving triangle
	return make_float4(barycentric, alpha);
}


// FIXME: add different masses
__device__ __forceinline__ void bounceParticleArray(
		int* validParticles, int nParticles,
		Particle v0, Particle v1, Particle v2,
		float3& f0, float3& f1, float3& f2,
		const float4* coosvels, int* collTime, float4* coosvels_new,
		float particleMass, float vertexMass, const float dt)
{
	const int tid = threadIdx.x % warpSize;
	const float threshold = 2e-6f;

	expandBy(v0.r, v1.r, v2.r, 2.0f*threshold);
	const float3 n = normalize(cross(v1.r-v0.r, v2.r-v0.r));

	for (int i=tid; i<nParticles; i+=warpSize)
	{
		const int pid = validParticles[i];
		Particle p(coosvels[2*pid], coosvels[2*pid+1]);

		float oldSign;
		float4 res = intersectParticleTriangleBarycentric(v0, v1, v2, n, p, dt, threshold, oldSign);
		float3 barycentricCoo = make_float3(res.x, res.y, res.z);
		float alpha = res.w;

		if (barycentricCoo.x >= 0.0f && barycentricCoo.y >= 0.0f && barycentricCoo.z >= 0.0f)
		{
			float alphaMin = __int_as_float( atomicMin(collTime+pid, __float_as_int(alpha)) );

			// A collision is found here
			const float3 vtri = barycentricCoo.x*v0.u + barycentricCoo.y*v1.u + barycentricCoo.z*v2.u;
			const float3 coo  = barycentricCoo.x*v0.r + barycentricCoo.y*v1.r + barycentricCoo.z*v2.r;

			triangleForces(v0.r, v1.r, v2.r, vertexMass, barycentricCoo, p.u - vtri, particleMass, dt, f0, f1, f2);

			float3 old = p.r;

			float3 newV = 2.0f*vtri-p.u;
			p.r = coo + threshold * n * ((oldSign > 0) ? 5.0f : -5.0f);
			p.u = newV;

//			if (p.i1 == 39464)
//				printf("BOUNCED from %d!! %f %f %f  ->  %f %f %f  (corr to %f). time %f, mintime %f\n",
//						(blockIdx.x * blockDim.x + threadIdx.x) / warpSize,
//						old.x, old.y, old.z, p.r.x, p.r.y, p.r.z,
//						dot(cross(v1.r-v0.r, v2.r-v0.r), p.r - v0.r),
//						alpha, alphaMin);

			if (alpha < alphaMin)
			{
				coosvels_new[2*pid]   = Float3_int(p.r, p.i1).toFloat4();
				coosvels_new[2*pid+1] = Float3_int(p.u, p.i2).toFloat4();
			}
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

//__launch_bounds__(128, 6)
__global__ void bounceMesh(
		const float4* coosvels, const uint* __restrict__ cellsStartSize, CellListInfo cinfo, int* collTime, float4* coosvels_new,
		const int nObj, const int nvertices, const int ntriangles, const int3* __restrict__ triangles, float4* objCoosvels, float* objForces,
		float particleMass, float vertexMass, const float dt)
{
	// About maximum distance a particle can cover in one step
	const float tol = 0.1f;

	// One warp per triangle
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int wid = gid / warpSize;
	const int tid = gid % warpSize;
	const int objId = wid / ntriangles;
	const int trid  = wid % ntriangles;
	if (objId >= nObj) return;

	// Preparation step. Filter out all the cells that don't intersect the triangle
	const int maxValidCells = 128;
	const int particleBufSize = 128;

	extern __shared__ int buffer[];
	int* count          = buffer + (threadIdx.x / warpSize);
	int* validCells     = buffer + (threadIdx.x / warpSize) * maxValidCells + blockDim.x / warpSize;
	int* validParticles = buffer + (threadIdx.x / warpSize) * particleBufSize + (blockDim.x / warpSize) * (maxValidCells + 1);

	*count = 0;

	const int3 triangle = triangles[trid];
	Particle v0 = Particle(objCoosvels, nvertices*objId + triangle.x);
	Particle v1 = Particle(objCoosvels, nvertices*objId + triangle.y);
	Particle v2 = Particle(objCoosvels, nvertices*objId + triangle.z);
	float3 n = cross(v1.r-v0.r, v2.r-v0.r);

	float3 lo = fminf(v0.r, fminf(v1.r, v2.r));
	float3 hi = fmaxf(v0.r, fmaxf(v1.r, v2.r));

	const int3 cidLow  = cinfo.getCellIdAlongAxis(lo - tol);
	const int3 cidHigh = cinfo.getCellIdAlongAxis(hi + tol);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	for (int i=tid; i<totCells+warpSize; i+=warpSize)
		if (i < totCells)
		{
			const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;

			float3 v000 = make_float3(cid3) * cinfo.h - cinfo.domainSize*0.5f;
			bool valid = isCellCrossingTriangle(v000, cinfo.h, n, v0.r, tol);

			if (valid)
			{
				int id = atomicAggInc(count);
				validCells[id] = cinfo.encode(cid3);
			}
		}


	// Second step. Make particle queue and process it
	const int nCells = *count;
	*count = 0;

	float3 f0 = make_float3(0.0f), f1 = make_float3(0.0f), f2 = make_float3(0.0f);
	for (int i=tid; i<nCells+warpSize; i+=warpSize)
	{
		if (i < -nCells)
		{
			int2 start_size = cinfo.decodeStartSize(cellsStartSize[validCells[i]]);

			int id = atomicAdd(count, start_size.y);
			for (int j = 0; j < start_size.y; j++)
				validParticles[id + j] = j + start_size.x;
		}

		if (*count >= particleBufSize/2)
		{
			bounceParticleArray(validParticles, particleBufSize/2, v0, v1, v2, f0, f1, f2,
					coosvels, collTime, coosvels_new, particleMass, vertexMass, dt);
			*count -= particleBufSize/2;

			for (int i=tid; i<particleBufSize/2; i+=warpSize)
				validParticles[i] = validParticles[i + particleBufSize/2];
		}
	}

	// Process remaining
	if (*count > 0)
	bounceParticleArray(validParticles, *count, v0, v1, v2, f0, f1, f2,
			coosvels, collTime, coosvels_new, particleMass, vertexMass, dt);

	auto plus = [] (float a, float b) {return a+b;};
	warpReduce(f0, plus);
	warpReduce(f1, plus);
	warpReduce(f2, plus);

	if (tid == 0)
	{
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
}

