#pragma once

#include <core/rbc_vector.h>
#include <core/cuda_common.h>

__global__ void computeAreaAndVolume(const float4* coosvels, MembraneMesh mesh, int nRbcs, float* areas, float* volumes)
{
	float2 a_v = make_float2(0.0f, 0.0f);
	const int cid = blockIdx.y;

	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < mesh.ntriangles; i += blockDim.x * gridDim.x)
	{
		int3 ids = mesh.triangles[i];

		float3 v0 = f4tof3( coosvels[ 2 * 3*(ids.x+cid*mesh.nvertices) ] );
		float3 v1 = f4tof3( coosvels[ 2 * 3*(ids.y+cid*mesh.nvertices) ] );
		float3 v2 = f4tof3( coosvels[ 2 * 3*(ids.z+cid*mesh.nvertices) ] );

		a_v.x += 0.5f * length(cross(v1 - v0, v2 - v0));
		a_v.y += 0.1666666667f * (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
				- v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
	}

	a_v = warpReduce( a_v, [] (float2 a, float2 b) { return a+b; } );
	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		atomicAdd(areas   + cid, a_v.x);
		atomicAdd(volumes + cid, a_v.y);
	}
}


// **************************************************************************************************
// **************************************************************************************************

__device__ __forceinline__ float3 _fangle(const float3 v1, const float3 v2, const float3 v3,
		const float area, const float volume, RBCvector::Parameters parameters)
{
	assert(parameters.q == 1);
	const float3 x21 = v2 - v1;
	const float3 x32 = v3 - v2;
	const float3 x31 = v3 - v1;

	const float3 normal = cross(x21, x31);

	const float n_2 = 2.0f * rsqrtf(dot(normal, normal));
	const float n_2to3 = n_2*n_2*n_2;
	const float coefArea = 0.25f * (parameters.Cq * n_2to3 -
			parameters.ka * (area - parameters.totArea0) * n_2);

	const float coeffVol = parameters.kv * (volume - parameters.totVolume0);
	const float3 addFArea = coefArea * cross(normal, x32);
	const float3 addFVolume = coeffVol * cross(v3, v2);

	float r = length(v2 - v1);
	r = r < 0.0001f ? 0.0001f : r;
	const float xx = r/parameters.lmax;
	const float IbforceI = parameters.kbToverp * ( 0.25f/((1.0f-xx)*(1.0f-xx)) - 0.25f + xx ) / r;

	return addFArea + addFVolume + IbforceI * x21;
}

__device__ __forceinline__ float3 _fvisc(const float3 v1, const float3 v2, const float3 u1, const float3 u2, RBCvector::Parameters parameters)
{
	const float3 du = u2 - u1;
	const float3 dr = v1 - v2;

	return du*parameters.gammaT + dr * parameters.gammaC*dot(du, dr) / dot(dr, dr);
}

template <int maxDegree>
__device__ float3 bondTriangleForce(Particle p, int locId, int rbcId, int nvertices,
		float4* const __restrict__ adjacent,
		float4* const __restrict__ coosvels,
		float* const __restrict__ areas, float* const __restrict__ volumes,
		RBCvector::Parameters parameters)
{
	const float3 r0 = p.r;
	const float3 u0 = p.u;

	const int startId = maxDegree * locId;
	int idv1 = adjacent[startId];
	Particle p1(coosvels, rbcId*nvertices + idv1);
	float3 r1 = p1.r;
	float3 u1 = p1.u;

	float3 f = make_float3(0.0f);

#pragma unroll 2
	for (int i=1; i<=maxDegree; i++)
	{
		int idv2 = adjacent[startId + (i % maxDegree)];
		if (idv2 == -1) break;

		Particle p2(coosvels, rbcId*nvertices + idv2);
		float3 r2 = p1.r;
		float3 u2 = p1.u;

		f += _fangle(r0, r1, r2, areas[rbcId], volumes[rbcId], parameters) +
				_fvisc(r0, r1, u0, u1, parameters);

		r1 = r2;
		u1 = u2;
	}

	return f;
}

// **************************************************************************************************

template<int update>
__device__  __forceinline__  float3 _fdihedral(float3 v1, float3 v2, float3 v3, float3 v4, RBCvector::Parameters parameters)
{
	const float3 ksi   = cross(v1 - v2, v1 - v3);
	const float3 dzeta = cross(v3 - v4, v2 - v4);

	const float overIksiI   = rsqrtf(dot(ksi, ksi));
	const float overIdzetaI = rsqrtf(dot(dzeta, dzeta));

	const float cosTheta = dot(ksi, dzeta) * overIksiI*overIdzetaI;
	const float IsinThetaI2 = 1.0f - cosTheta*cosTheta;
	const float sinTheta_1 = copysignf( rsqrtf(max(IsinThetaI2, 1.0e-6f)), dot(ksi - dzeta, v4 - v1) );  // ">" because the normals look inside
	const float beta = parameters.cost0kb - cosTheta * parameters.sint0kb * sinTheta_1;

	float b11 = -beta * cosTheta * overIksiI*overIksiI;
	float b12 = beta * overIksiI*overIdzetaI;
	float b22 = -beta * cosTheta * overIdzetaI*overIdzetaI;

	if (update == 1)
		return cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;
	else if (update == 2)
		return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
	else return make_float3(0, 0, 0);
}


template <int maxDegree>
__device__
float3 dihedralForce(Particle p, int locId, int rbcId, int nvertices,
		float4* const __restrict__ adjacent, int* const __restrict__ adjacent_second,
		float4* const __restrict__ coosvels,
		RBCvector::Parameters parameters)
{
	const int shift = 2*rbcId*nvertices;
	const float3 r0 = p.r;

	const int startId = maxDegree * locId;
	int idv1 = adjacent[startId];
	int idv2 = adjacent[startId+1];

	float3 r1 = Float3_int(coosvels[shift + 2*idv1]).v;
	float3 r2 = Float3_int(coosvels[shift + 2*idv2]).v;

	float3 f = make_float3(0.0f);

	//       v4
	//     /   \
	//   v1 --> v2 --> v3
	//     \   /
	//       V
	//       v0

	// dihedrals: 0124, 0123

#pragma unroll 2
	for (int i=1; i<=maxDegree; i++)
	{
		int idv3 = adjacent       [startId + ( (i+1) % maxDegree )];
		int idv4 = adjacent_second[startId + (  i    % maxDegree )];

		if (idv3 == -1 && idv4 == -1) break;

		float3 r3, r4;
		if (idv3 != -1) r3 = Float3_int(coosvels[shift + 2*idv3]).v;
		r4 =				 Float3_int(coosvels[shift + 2*idv4]).v;


		f +=  _fdihedral<1>(r0, r2, r1, r4, parameters);
		if (idv3 != -1)
			f+= _fdihedral<2>(r1, r0, r2, r3, parameters);

		r1 = r2;
		r2 = r3;
	}

	return f;
}

template <int maxDegree>
__global__ __launch_bounds__(128, 12)
void computeMembraneForces(float4* const __restrict__ adjacent, int* const __restrict__ adjacent_second, int nvertices,
		const int nrbcs, float4* const __restrict__ coosvels, float* const __restrict__ areas, float* const __restrict__ volumes, float* forces,
		RBCvector::Parameters parameters)
{
	const int pid = threadIdx.x + blockDim.x * blockIdx.x;
	const int locId = pid % nvertices;
	const int rbcId = pid / nvertices;

	if (pid >= nrbcs*nvertices) return;

	Particle p(coosvels, pid);

	float3 f = bondTriangleForce<maxDegree>(p, locId, rbcId, nvertices, adjacent, coosvels, areas, volumes, parameters);
	f += dihedralForce<maxDegree>(p, locId, rbcId, nvertices, adjacent, adjacent_second, coosvels, parameters);

	atomicAdd(forces + 4*pid, f);
}





