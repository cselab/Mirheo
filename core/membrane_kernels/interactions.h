#pragma once

#include <core/utils/cuda_common.h>
#include <core/pvs/object_vector.h>
#include <core/mesh.h>

struct GPU_RBCparameters
{
	float gammaC, gammaT;
	float mpow, l0, lmax, lmax_1, kbT_over_p_lmax;
	float area0, totArea0, totVolume0;
	float cost0kb, sint0kb;
	float ka0, kv0, kd0;
};

__global__ void computeAreaAndVolume(OVviewWithAreaVolume view, MeshView mesh)
{
	const int objId = blockIdx.x;
	float2 a_v = make_float2(0.0f);

	for(int i = threadIdx.x; i < mesh.ntriangles; i += blockDim.x)
	{
		int3 ids = mesh.triangles[i];

		float3 v0 = f4tof3( view.particles[ 2 * (ids.x+objId*mesh.nvertices) ] );
		float3 v1 = f4tof3( view.particles[ 2 * (ids.y+objId*mesh.nvertices) ] );
		float3 v2 = f4tof3( view.particles[ 2 * (ids.z+objId*mesh.nvertices) ] );

		a_v.x += 0.5f * length(cross(v1 - v0, v2 - v0));
		a_v.y += 0.1666666667f * (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
				- v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
	}

	a_v = warpReduce( a_v, [] (float a, float b) { return a+b; } );
	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		atomicAdd(&view.area_volumes[objId], a_v);
	}
}


// **************************************************************************************************
// **************************************************************************************************

__device__ inline float fastPower(const float x, const float k)
{
	if (fabsf(k - 3.0f) < 1e-6f) return x*x*x;
	if (fabsf(k - 2.0f) < 1e-6f) return x*x;
	if (fabsf(k - 1.0f) < 1e-6f) return x;
	if (fabsf(k - 0.5f) < 1e-6f) return sqrtf(fabsf(x));

    return powf(fabsf(x), k);
}


__device__ inline float3 _fangle(const float3 v1, const float3 v2, const float3 v3,
		const float totArea, const float totVolume, GPU_RBCparameters parameters)
{
	const float3 x21 = v2 - v1;
	const float3 x32 = v3 - v2;
	const float3 x31 = v3 - v1;

	const float3 normal = cross(x21, x31);

	const float area = 0.5f * length(normal);
	const float area_1 = 1.0f / area;

	// TODO: optimize computations here
	const float coefArea = -0.25f * (
			parameters.ka0 * (totArea - parameters.totArea0) * area_1
			+ parameters.kd0 * (area - parameters.area0) / (area * parameters.area0) );

	const float coeffVol = parameters.kv0 * (totVolume - parameters.totVolume0);
	const float3 fArea = coefArea * cross(normal, x32);
	const float3 fVolume = coeffVol * cross(v3, v2);

	return fArea + fVolume;
}

__device__ inline float3 _fbond(const float3 v1, const float3 v2, const float l0, GPU_RBCparameters parameters)
{
	float r = max(length(v2 - v1), 1e-5f);

	auto wlc = [parameters] (float x) {
		return parameters.kbT_over_p_lmax * (4.0f*x*x - 9.0f*x + 6.0f) / ( 4.0f*sqr(1.0f - x) );
	};

	const float IbforceI_wlc = wlc( min(parameters.lmax - 1e-6f, r) * parameters.lmax_1 );

	const float kp = wlc( l0 * parameters.lmax_1 ) * fastPower(l0, parameters.mpow+1);

	const float IbforceI_pow = -kp / (fastPower(r, parameters.mpow+1));

	const float IfI = min(200.0f, max(-200.0f, IbforceI_wlc + IbforceI_pow));

	return IfI * (v2 - v1);
}

__device__ inline float3 _fvisc(Particle p1, Particle p2, GPU_RBCparameters parameters)
{
	const float3 du = p2.u - p1.u;
	const float3 dr = p1.r - p2.r;

	return du*parameters.gammaT + dr * parameters.gammaC*dot(du, dr) / dot(dr, dr);
}

template <int maxDegree>
__device__ float3 bondTriangleForce(
		Particle p, int locId, int rbcId,
		const OVviewWithAreaVolume& view,
		const MembraneMeshView& mesh,
		const GPU_RBCparameters& parameters)
{
	float3 f = make_float3(0.0f);
	const int startId = maxDegree * locId;
	const int degree = mesh.degrees[locId];

	int idv1 = mesh.adjacent[startId];
	Particle p1(view.particles, rbcId*mesh.nvertices + idv1);

#pragma unroll 2
	for (int i=1; i<=degree; i++)
	{
		int idv2 = mesh.adjacent[startId + (i % degree)];

		Particle p2(view.particles, rbcId*mesh.nvertices + idv2);

		f += _fangle(p.r, p1.r, p2.r, view.area_volumes[rbcId].x, view.area_volumes[rbcId].y, parameters) +
			 _fbond(p.r, p1.r, mesh.initialLengths[startId + i-1], parameters) +  _fvisc (p, p1, parameters);

		idv1 = idv2;
		p1 = p2;
	}

	return f;
}

// **************************************************************************************************

template<int update>
__device__  inline  float3 _fdihedral(float3 v1, float3 v2, float3 v3, float3 v4, GPU_RBCparameters parameters)
{
	const float3 ksi   = cross(v1 - v2, v1 - v3);
	const float3 dzeta = cross(v3 - v4, v2 - v4);

	const float overIksiI   = rsqrtf(dot(ksi, ksi));
	const float overIdzetaI = rsqrtf(dot(dzeta, dzeta));

	const float cosTheta = dot(ksi, dzeta) * overIksiI * overIdzetaI;
	const float IsinThetaI2 = 1.0f - cosTheta*cosTheta;

	const float rawST_1 = rsqrtf(max(IsinThetaI2, 1.0e-6f));
	const float sinTheta_1 = copysignf( rawST_1, dot(ksi - dzeta, v4 - v1) ); // because the normals look inside
	const float beta = parameters.cost0kb - cosTheta * parameters.sint0kb * sinTheta_1;

	float b11 = -beta * cosTheta *  overIksiI   * overIksiI;
	float b12 =  beta *             overIksiI   * overIdzetaI;
	float b22 = -beta * cosTheta *  overIdzetaI * overIdzetaI;

	if (update == 1)
		return cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;
	else if (update == 2)
		return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
	else return make_float3(0.0f);
}

template <int maxDegree>
__device__ float3 dihedralForce(
		Particle p, int locId, int rbcId,
		const OVviewWithAreaVolume& view,
		const MembraneMeshView& mesh,
		const GPU_RBCparameters& parameters)
{
	const int shift = 2*rbcId*mesh.nvertices;
	const float3 r0 = p.r;

	const int startId = maxDegree * locId;
	const int degree = mesh.degrees[locId];

	int idv1 = mesh.adjacent[startId];
	int idv2 = mesh.adjacent[startId+1];

	float3 r1 = Float3_int(view.particles[shift + 2*idv1]).v;
	float3 r2 = Float3_int(view.particles[shift + 2*idv2]).v;

	float3 f = make_float3(0.0f);

	//       v4
	//     /   \
	//   v1 --> v2 --> v3
	//     \   /
	//       V
	//       v0

	// dihedrals: 0124, 0123


#pragma unroll 2
	for (int i=0; i<degree; i++)
	{
		int idv3 = mesh.adjacent       [startId + (i+2) % degree];
		int idv4 = mesh.adjacent_second[startId + i];

		float3 r3, r4;
		r3 = Float3_int(view.particles[shift + 2*idv3]).v;
		r4 = Float3_int(view.particles[shift + 2*idv4]).v;

		f += _fdihedral<1>(r0, r2, r1, r4, parameters);
		f += _fdihedral<2>(r1, r0, r2, r3, parameters);

		r1 = r2;
		r2 = r3;

		idv1 = idv2;
		idv2 = idv3;
	}

	return f;
}

template <int maxDegree>
//__launch_bounds__(128, 12)
__global__ void computeMembraneForces(
		OVviewWithAreaVolume view,
		MembraneMeshView mesh,
		GPU_RBCparameters parameters)
{
	// RBC particles are at the same time mesh vertices
	assert(view.objSize == mesh.nvertices);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;
	const int locId = pid % mesh.nvertices;
	const int rbcId = pid / mesh.nvertices;

	if (pid >= view.nObjects * mesh.nvertices) return;

//	if (locId == 0)
//		printf("%d: area %f  volume %f\n", rbcId, view.area_volumes[rbcId].x, view.area_volumes[rbcId].y);

	Particle p(view.particles, pid);

	float3 f = bondTriangleForce<maxDegree>(p, locId, rbcId, view, mesh, parameters)
			 + dihedralForce    <maxDegree>(p, locId, rbcId, view, mesh, parameters);

	atomicAdd(view.forces + pid, f);
}





