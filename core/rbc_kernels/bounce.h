#pragma once

#include <core/rbc_vector.h>
#include <core/cuda_common.h>
#include <core/celllist.h>
#include <core/bounce_solver.h>


// Expand the triangle by extra for robustness
__device__ __forceinline__ void expandBy(float3& r0, float3& r1, float3& r2, float extra)
{
	auto invLen = [] (float3 x) {
		return rsqrt(dot(x, x));
	};

	float3 center = 0.33333333333f * (r0 + r1 + r2);
	r0 = (r0 - center) * (1.0f + extra * invLen(r0-center)) + center;
	r1 = (r1 - center) * (1.0f + extra * invLen(r1-center)) + center;
	r2 = (r2 - center) * (1.0f + extra * invLen(r2-center)) + center;
}


// https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Determining_location_with_respect_to_a_triangle
__device__ __forceinline__ float3 computeBaricentric(float3 r0, float3 r1, float3 r2, float3 v)
{
	float denominator = 1.0f / ( (r1.y-r2.y)*(r0.x-r2.x) + (r2.x-r1.x)*(r0.y-r2.y) );

	float l0 = (r1.y-r2.y)*(v.x-r2.x) + (r2.x-r1.x)*(v.y-r2.y);
	float l1 = (r2.y-r0.y)*(v.x-r2.x) + (r0.x-r2.x)*(v.y-r2.y);
	float l2 = 1.0f - l0 - l1;

	return make_float3(l0, l1, l2);
}

// Particle with mass M and velocity U hits triangle (x0, x1, x2)
// into point O. Vertex masses are m. Treated as rigid and stationary,
// what are the vertex forces induced by the collision?
__device__ __forceinline__ void triangleForces(
		float3 x0, float3 x1, float3 x2, float m,
		float3 O_baricentric, float3 U, float M,
		float dt,
		float3& f0, float3& f1, float3& f2)
{
	auto invLen = [] (float3 x) {
		return rsqrt(dot(x, x));
	};

	const float3 n = cross(x1-x0, x2-x0);

	const float IU_ortI = dot(U, n);
	const float3 U_par = U - IU_ortI * n;

	const float a = 2.0f*M/m * IU_ortI;
	const float v0_ort = O_baricentric.x * a;
	const float v1_ort = O_baricentric.y * a;
	const float v2_ort = O_baricentric.z * a;

	const float3 C = 0.333333333f * (x0+x1+x2);
	const float3 Vc = 0.666666666f * M/m * U_par;

	const float Ir0I_1 = invLen(C-x0);
	const float Ir1I_1 = invLen(C-x1);
	const float Ir2I_1 = invLen(C-x2);

	const float3 O = O_baricentric.x * x0 + O_baricentric.x * x1 + O_baricentric.x * x2;
	const float3 L = 2.0f*M * cross(C-O, U_par);
	const float b = dot(L, n)/m;

	const float3 normal2r0 = normalize(cross(C-x0, n));
	const float3 normal2r1 = normalize(cross(C-x1, n));
	const float3 normal2r2 = normalize(cross(C-x2, n));

	const float3 u0 = b * Ir0I_1 * normal2r0;
	const float3 u1 = b * Ir1I_1 * normal2r1;
	const float3 u2 = b * Ir2I_1 * normal2r2;

	const float3 v0 = v0_ort*n + Vc + u0;
	const float3 v1 = v1_ort*n + Vc + u1;
	const float3 v2 = v2_ort*n + Vc + u2;

	const float invdt = 1.0f / dt;
	f0 += v0 * m * invdt;
	f1 += v1 * m * invdt;
	f2 += v2 * m * invdt;
}

// find baricentric coordinates of the collision
// if at least one of the value returned negative there was no collision
__device__ __forceinline__ float3 intersectParticleTriangleBaricentric(
		Particle v0, Particle v1, Particle v2, float3 n,
		Particle p, float dt, float threshold)
{
	// Increase the triangle a bit for robustness
	expandBy(v0.r, v1.r, v2.r, threshold);

	auto F = [=] (float lambda) {
		float t = (1.0f - lambda)*dt;
		float3 r0 = v0.r - v0.u*t;
		float3 r1 = v1.r - v1.u*t;
		float3 r2 = v2.r - v2.u*t;

		float3 x  = p .r - p .u*t;

		return dot(cross(r1-r0, r2-r0), x - r0);
	};

	float vold = F(0.0f);
	float vnew = F(1.0f);

	// Particle is far
	// No need for baricentric coordinates, skip it
	if ( fabs(vnew) > threshold ) return make_float3(-1000.0f);

	float3 projected;
	float alpha = 1.0f;

	// Particle is in a near band but it didn't cross the triangle
	if (vnew*vold >= 0.0f)
		projected = p.r - dot(p.r-v0.r, n)*n;
	else
	{
		// Particle has crossed the triangle plane
		alpha = solveLinSearch(F);

		// This should NEVER happen
		if (alpha < -0.1f)
			printf("Something awful happened with mesh bounce");

		// This is the collision point with the MOVING triangle
		float3 projected = v0.r - v0.u*(1.0f - alpha)*dt;
	}

	// The triangle with which the collision was detected has a timestamp (1.0f - alpha)*dt
	float t = (1.0f - alpha)*dt;
	float3 r0 = v0.r - v0.u*t;
	float3 r1 = v1.r - v1.u*t;
	float3 r2 = v2.r - v2.u*t;

	// Return the baricentric coordinates of the projected position
	// on the corresponding moving triangle
	return computeBaricentric(r0, r1, r2, projected);
}


// FIXME: add different masses
__device__ __forceinline__ void bounceParticleArray(
		int* validParticles, int nParticles,
		Particle v0, Particle v1, Particle v2,
		float3& f0, float3& f1, float3& f2,
		float4* coosvels, float mass, const float dt)
{
	const int tid = threadIdx.x / warpSize;
	const float threshold = 2e-6f;

	expandBy(v0.r, v1.r, v2.r, 2.0f*threshold);
	const float3 n = normalize(cross(v1.r-v0.r, v2.r-v0.r));

	for (int i=tid; i<nParticles; i+=warpSize)
	{
		const int pid = validParticles[i];
		Particle p(coosvels[2*pid], coosvels[2*pid+1]);

		float3 baricentricCoo = intersectParticleTriangleBaricentric(v0, v1, v2, n, p, dt, threshold);
		if (baricentricCoo.x >= 0.0f && baricentricCoo.y >= 0.0f && baricentricCoo.z >= 0.0f)
		{
			// A collision is found here
			const float3 vtri = baricentricCoo.x*v0.u + baricentricCoo.y*v1.u + baricentricCoo.z*v2.u;
			const float3 coo  = baricentricCoo.x*v0.r + baricentricCoo.y*v1.r + baricentricCoo.z*v2.r;

			triangleForces(v0.r, v1.r, v2.r, mass, baricentricCoo, p.u - vtri, mass, dt, f0, f1, f2);

			float3 newV = 2.0f*vtri-p.u;
			const float s = dot(p.r - coo, n);
			p.r = coo + s*2.0f*threshold;
			p.u = newV;

			coosvels[2*pid]   = Float3_int(p.r, p.i1).toFloat4();
			coosvels[2*pid+1] = Float3_int(p.u, p.i2).toFloat4();
		}
	}
}

//__launch_bounds__(128, 7)
__global__ void bounceMesh(
		float4* coosvels, float mass, const uint* __restrict__ cellsStartSize, CellListInfo cinfo,
		const int nObj, const int nvertices, const int ntriangles, const int3* __restrict__ triangles, float4* objCoosvels, float* objForces,
		const float dt)
{
	const float threshold = 0.2f;

	// One warp per triangle
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int wid = gid / warpSize;
	const int tid = gid % warpSize;
	const int objId = wid / ntriangles;
	const int trid  = wid % ntriangles;
	if (objId >= nObj) return;

	// Preparation step. Filter out all the cells that don't intersect the triangle
	const int maxValidCells = 64;
	const int particleBufSize = 128;

	extern __shared__ int buffer[];
	__shared__ int* count          = buffer + (threadIdx.x / warpSize);
	__shared__ int* validCells     = buffer + (threadIdx.x / warpSize) * maxValidCells + blockDim.x / warpSize;
	__shared__ int* validParticles = buffer + (threadIdx.x / warpSize) * particleBufSize + (blockDim.x / warpSize) * (maxValidCells + 1);

	*count = 0;

	const int3 triangle = triangles[trid];
	Particle v0 = Particle(objCoosvels, nvertices*objId + triangle.x);
	Particle v1 = Particle(objCoosvels, nvertices*objId + triangle.y);
	Particle v2 = Particle(objCoosvels, nvertices*objId + triangle.z);
	float3 n = cross(v1.r-v0.r, v2.r-v0.r);

	float3 lo = fminf(v0.r, fminf(v1.r, v2.r));
	float3 hi = fmaxf(v0.r, fmaxf(v1.r, v2.r));

	const int3 cidLow  = cinfo.getCellIdAlongAxis(lo - 0.5f);
	const int3 cidHigh = cinfo.getCellIdAlongAxis(hi + 0.5f);

	const int3 span = cidHigh - cidLow + make_int3(1,1,1);
	const int totCells = span.x * span.y * span.z;

	for (int i=tid; i<totCells+warpSize; i+=warpSize)
	{
		if (i < totCells)
		{
			const int3 cid3 = make_int3( i % span.x, (i/span.x) % span.y, i / (span.x*span.y) ) + cidLow;

			float3 v000 = make_float3(cid3) * cinfo.h - cinfo.domainSize*0.5f;

			float3 v001 = v000 + make_float3(        0,         0, cinfo.h.z);
			float3 v010 = v000 + make_float3(        0, cinfo.h.y,         0);
			float3 v011 = v000 + make_float3(        0, cinfo.h.y, cinfo.h.z);
			float3 v100 = v000 + make_float3(cinfo.h.x,         0,         0);
			float3 v101 = v000 + make_float3(cinfo.h.x,         0, cinfo.h.z);
			float3 v110 = v000 + make_float3(cinfo.h.x, cinfo.h.y,         0);
			float3 v111 = v000 + make_float3(cinfo.h.x, cinfo.h.y, cinfo.h.z);

			int pos = 0;
			int neg = 0;
			float tmp;

			tmp = dot(n, v000-v0.r);
			pos += ( tmp > -threshold );
			neg += ( tmp <  threshold );

			tmp = dot(n, v001-v0.r);
			pos += ( tmp > -threshold );
			neg += ( tmp <  threshold );

			tmp = dot(n, v010-v0.r);
			pos += ( tmp > -threshold );
			neg += ( tmp <  threshold );

			tmp = dot(n, v011-v0.r);
			pos += ( tmp > -threshold );
			neg += ( tmp <  threshold );

			tmp = dot(n, v100-v0.r);
			pos += ( tmp > -threshold );
			neg += ( tmp <  threshold );

			tmp = dot(n, v101-v0.r);
			pos += ( tmp > -threshold );
			neg += ( tmp <  threshold );

			tmp = dot(n, v110-v0.r);
			pos += ( tmp > -threshold );
			neg += ( tmp <  threshold );

			tmp = dot(n, v111-v0.r);
			pos += ( tmp > -threshold );
			neg += ( tmp <  threshold );


			if (pos != 8 && neg != 8)
			{
				int id = atomicAggInc(count);
				validCells[id] = cinfo.encode(cid3);
			}
		}
	}


	// Second step. Make particle queue and process it
	const int nCells = count;
	*count = 0;

	float3 f0 = 0.0f, f1 = 0.0f, f2 = 0.0f;
	for (int i=tid; i<nCells; i+=warpSize)
	{
		int2 start_size = cinfo.decodeStartSize(cellsStartSize[validCells[i]]);

		int id = atomicAdd(count, start_size.y);
		for (int j = 0; j < start_size.y; j++)
			validParticles[id + j] = j + start_size.x;

		if (count >= particleBufSize/2)
		{
			bounceParticleArray(validParticles, particleBufSize/2, v0, v1, v2, f0, f1, f2, coosvels, mass, dt);
			count -= particleBufSize/2;

			for (int i=tid; i<particleBufSize/2; i+=warpSize)
				validParticles[i] = validParticles[i + particleBufSize/2];
		}
	}


	// Process remaining
	bounceParticleArray(validParticles, *count, v0, v1, v2, f0, f1, f2, coosvels, mass, dt);

	auto plus = [] (float3 a, float3 b) {return a+b;};
	warpReduce(f0, plus);
	warpReduce(f1, plus);
	warpReduce(f2, plus);

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
