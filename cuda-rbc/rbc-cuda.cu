/*
 *  rbc.cpp
 *  ctc local
 *
 *  Created by Dmitry Alexeev on Nov 3, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */


#include <math_functions.h>
#include <cstdio>
#include <map>
#include <string>
#include <fstream>
#include <iostream>

#include "rbc-cuda.h"
#include "vec3.h"
#include "cuda-common.h"

using namespace std;

namespace CudaRBC
{

int nparticles;
int ntriang;
int nbonds;
int ndihedrals;

int *triangles_host;
int *triangles;
int *bonds;
int *dihedrals;

// Helper pointers
real *totA_V;
int  *mapping;

// Original configuration
real* orig_xyzuvw;

map<cudaStream_t, float*> bufmap;
__constant__ float A[4][4];


void setup(int& nvertices, Extent& host_extent)
{
	const bool report = true;

	const char* fname = "../cuda-rbc/rbc2.atom_parsed";
	ifstream in(fname);
	string line;

	if (report)
		if (in.good())
		{
			cout << "Reading file " << fname << endl;
		}
		else
		{
			cout << fname << ": no such file" << endl;
			exit(1);
		}

	in >> nparticles >> nbonds >> ntriang >> ndihedrals;

	if (report)
		if (in.good())
		{
			cout << "File contains " << nparticles << " atoms, " << nbonds << " bonds, " << ntriang << " triangles and " << ndihedrals << " dihedrals" << endl;
		}
		else
		{
			cout << "Couldn't parse the file" << endl;
			exit(1);
		}

	// Atoms section
	real *xyzuvw_host = new real[6*nparticles];

	int cur = 0;
	int tmp1, tmp2, aid;
	while (in.good() && cur < nparticles)
	{
		in >> tmp1 >> tmp2 >> aid >> xyzuvw_host[6*cur+0] >> xyzuvw_host[6*cur+1] >> xyzuvw_host[6*cur+2];
		xyzuvw_host[6*cur+3] = xyzuvw_host[6*cur+4] = xyzuvw_host[6*cur+5] = 0;
		if (aid != 1) break;
		cur++;
	}

	// Shift the origin of "zeroth" rbc to 0,0,0
	float xmin[3] = { 1e10,  1e10,  1e10};
	float xmax[3] = {-1e10, -1e10, -1e10};

	for (int i=0; i<cur; i++)
		for (int d=0; d<3; d++)
		{
			xmin[d] = min(xmin[d], xyzuvw_host[6*i + d]);
			xmax[d] = max(xmax[d], xyzuvw_host[6*i + d]);
		}

	float origin[3];
	for (int d=0; d<3; d++)
		origin[d] = 0.5 * (xmin[d] + xmax[d]);

	for (int i=0; i<cur; i++)
		for (int d=0; d<3; d++)
			xyzuvw_host[6*i + d] -= origin[d];

	int *used = new int[nparticles];
	int *mapping_host = new int[4*(ntriang + ndihedrals)];
	memset(used, 0, nparticles*sizeof(int));
	memset(mapping_host, 0, 4*(ntriang + ndihedrals)*sizeof(int));
	int id0, id1, id2, id3;

	// Bonds section

	int *bonds_host = new int[nbonds * 2];
	for (int i=0; i<nbonds; i++)
	{
		in >> tmp1 >> tmp2 >> id0 >> id1;
		id0--; id1--;
		bonds_host[2*i + 0] = id0;
		bonds_host[2*i + 1] = id1;
	}

	// Angles section --> triangles

	triangles_host = new int[3*ntriang];
	for (int i=0; i<ntriang; i++)
	{
		in >> tmp1 >> tmp2 >> id0 >> id1 >> id2;

		id0--; id1--; id2--;
		triangles_host[3*i + 0] = id0;
		triangles_host[3*i + 1] = id1;
		triangles_host[3*i + 2] = id2;

		mapping_host[4*i + 0] = (used[id0]++);
		mapping_host[4*i + 1] = (used[id1]++);
		mapping_host[4*i + 2] = (used[id2]++);
	}

	// Dihedrals section

	int *dihedrals_host = new int[4*ndihedrals];
	for (int i=0; i<ndihedrals; i++)
	{
		in >> tmp1 >> tmp2 >> id0 >> id1 >> id2 >> id3;
		id0--; id1--; id2--; id3--;

		dihedrals_host[4*i + 0] = id0;
		dihedrals_host[4*i + 1] = id1;
		dihedrals_host[4*i + 2] = id2;
		dihedrals_host[4*i + 3] = id3;

		mapping_host[4*(i + ntriang) + 0] = (used[id0]++);
		mapping_host[4*(i + ntriang) + 1] = (used[id1]++);
		mapping_host[4*(i + ntriang) + 2] = (used[id2]++);
		mapping_host[4*(i + ntriang) + 3] = (used[id3]++);
	}

	in.close();

	gpuErrchk( cudaMalloc(&orig_xyzuvw, nparticles             * 6 * sizeof(float)) );
	gpuErrchk( cudaMalloc(&triangles,   ntriang                * 3 * sizeof(int)) );
	gpuErrchk( cudaMalloc(&dihedrals,   ndihedrals             * 4 * sizeof(int)) );
	gpuErrchk( cudaMalloc(&mapping,     (ndihedrals + ntriang) * 4 * sizeof(int)) );

	gpuErrchk( cudaMemcpy(orig_xyzuvw, xyzuvw_host,    nparticles             * 6 * sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(triangles,   triangles_host, ntriang                * 3 * sizeof(int),   cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(dihedrals,   dihedrals_host, ndihedrals             * 4 * sizeof(int),   cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(mapping,     mapping_host,   (ndihedrals + ntriang) * 4 * sizeof(int),   cudaMemcpyHostToDevice) );

	delete[] xyzuvw_host;
	delete[] dihedrals_host;
	delete[] mapping_host;

	nvertices = nparticles;
	host_extent.xmin = xmin[0] - origin[0];
	host_extent.ymin = xmin[1] - origin[1];
	host_extent.zmin = xmin[2] - origin[2];

	host_extent.xmax = xmax[0] - origin[0];
	host_extent.ymax = xmax[1] - origin[1];
	host_extent.zmax = xmax[2] - origin[2];

	gpuErrchk( cudaMalloc(&totA_V, 2 * sizeof(float)) );

	//        0.0945, 0.00141, 1.642599,
	//        1, 1.8, a, v, a/m.ntriang, 945, 0, 472.5,
	//        90, 30, sin(phi), cos(phi), 6.048

	params.kbT = 0.0945;
	params.p = 0.00141;
	params.lmax = 1.642599;
	params.q = 1;
	params.Cq = 1.8;
	params.totArea0 = 135;
	params.totVolume0 = 94;
	params.area0 = params.totArea0 / (float)ntriang;
	params.ka = 945;
	params.kd = 10;
	params.kv = 150;
	params.gammaT = 90;
	params.gammaC = 30;

	float phi = 6.9 / 180.0*M_PI;
	params.sinTheta0 = sin(phi);
	params.cosTheta0 = cos(phi);
	params.kb = 6.048;
}

    int get_nvertices() { return nparticles; }
    
__global__ void transformKernel(float* xyzuvw, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	float x = xyzuvw[6*i + 0];
	float y = xyzuvw[6*i + 1];
	float z = xyzuvw[6*i + 2];

	xyzuvw[6*i + 0] = A[0][0]*x + A[0][1]*y + A[0][2]*z + A[0][3];
	xyzuvw[6*i + 1] = A[1][0]*x + A[1][1]*y + A[1][2]*z + A[1][3];
	xyzuvw[6*i + 2] = A[2][0]*x + A[2][1]*y + A[2][2]*z + A[2][3];
}

void initialize(float *device_xyzuvw, const float (*transform)[4])
{
	const int threads = 128;
	const int blocks  = (nparticles + threads - 1) / threads;

	gpuErrchk( cudaMemcpyToSymbol(A, transform, 16 * sizeof(float)) );
	gpuErrchk( cudaMemcpy(device_xyzuvw, orig_xyzuvw, 6*nparticles * sizeof(float), cudaMemcpyDeviceToDevice) );
	transformKernel<<<blocks, threads>>>(device_xyzuvw, nparticles);
}

__inline__ __host__ __device__ float3 fminf(float3 a, float3 b)
{
	return make_float3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

__inline__ __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
	return make_float3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

__device__ float atomicMin(float *addr, float value)
{
	float old = *addr, assumed;
	if(old <= value) return old;

	do
	{
		assumed = old;
		old = __int_as_float( atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(min(value, assumed))) );
	}while(old!=assumed);

	return old;
}

__device__ float atomicMax(float *addr, float value)
{
	float old = *addr, assumed;
	if(old >= value) return old;

	do
	{
		assumed = old;
		old = __int_as_float( atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(max(value, assumed))) );
	}while(old!=assumed);

	return old;
}


__global__ void extentKernel(const float* const xyzuvw, Extent* extent, int npart)
{
	float3 loBound = make_float3( 1e10f,  1e10f,  1e10f);
	float3 hiBound = make_float3(-1e10f, -1e10f, -1e10f);

	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < npart; i += blockDim.x * gridDim.x)
	{
		float3 v = make_float3(xyzuvw[6*i + 0], xyzuvw[6*i + 1], xyzuvw[6*i + 2]);

		loBound = fminf(loBound, v);
		hiBound = fmaxf(hiBound, v);
	}

	loBound = warpReduceMin(loBound);
	__syncthreads();
	hiBound = warpReduceMax(hiBound);

	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		atomicMin(&extent->xmin, loBound.x);
		atomicMin(&extent->ymin, loBound.y);
		atomicMin(&extent->zmin, loBound.z);

		atomicMax(&extent->xmax, hiBound.x);
		atomicMax(&extent->ymax, hiBound.y);
		atomicMax(&extent->zmax, hiBound.z);
	}
}

void extent_nohost(cudaStream_t stream, const float * const xyzuvw, Extent * device_extent)
{
	const int threads = 128;
	const int blocks  = (nparticles + threads - 1) / threads;

	Extent dummy;
	dummy.xmin = dummy.ymin = dummy.zmin = 1e10;
	dummy.xmax = dummy.ymax = dummy.zmax = -1e10;
	gpuErrchk( cudaMemcpy(device_extent, &dummy, sizeof(Extent), cudaMemcpyHostToDevice) );

	extentKernel<<<blocks, threads, 0, stream>>>(xyzuvw, device_extent, nparticles);
}


__global__ void areaAndVolumeKernel(int* triangles, const float* const xyzuvw, float* totA_V, int ntriang)
{
	float2 a_v = make_float2(0.0f, 0.0f);

	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < ntriang; i += blockDim.x * gridDim.x)
	{
		int id0 = triangles[3*i + 0];
		vec3 v0(xyzuvw+6*id0);

		int id1 = triangles[3*i + 1];
		vec3 v1(xyzuvw+6*id1);

		int id2 = triangles[3*i + 2];
		vec3 v2(xyzuvw+6*id2);

		a_v.x += 0.5f * norm(cross(v1 - v0, v2 - v0));
		a_v.y += 0.1666666667f * (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
				- v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
	}

	a_v = warpReduceSum(a_v);
	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		atomicAdd(&totA_V[0], a_v.x);
		atomicAdd(&totA_V[1], a_v.y);
	}
}


// Assume that values are already transferred to device
__host__ void areaAndVolume(const float* const xyzuvw)
{
	const int threads = 128;
	const int blocks  = (ntriang + threads - 1) / threads;

	gpuErrchk( cudaMemset(totA_V, 0, 2 * sizeof(float)) );
	areaAndVolumeKernel<<<blocks, threads>>>(triangles, xyzuvw, totA_V, ntriang);
}


__global__ void perTriangle(int* triangles, const float* const xyzuvw, float* fxfyfz, int* mapping, int ntriang,
		float q, float Cq, float* totA_V, float totArea0, float totVolume0, float area0, float ka, float kd, float kv)
{
	float totArea = totA_V[0];
	float totVolume = totA_V[1];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ntriang) return;

	int id0 = triangles[3*i + 0];
	vec3 v0(xyzuvw+6*id0);

	int id1 = triangles[3*i + 1];
	vec3 v1(xyzuvw+6*id1);

	int id2 = triangles[3*i + 2];
	vec3 v2(xyzuvw+6*id2);

	vec3 ksi = cross(v1 - v0, v2 - v0);
	float area = 0.5f * norm(ksi);

	// in-plane
	float alpha = 0.25f * q*Cq / powf(area, q+2.0);

	// area conservation
	float beta_a = -0.25f * ( ka*(totArea - totArea0) / (totArea0*area) + kd * (area - area0) / (area0 * area) );
	alpha += beta_a;
	vec3 f0, f1, f2;

	f0 = cross(ksi, v2-v1)*alpha;
	f1 = cross(ksi, v0-v2)*alpha;
	f2 = cross(ksi, v1-v0)*alpha;

	// volume conservation
	// "-" here is because the normal look inside
	vec3 ksi_3 = ksi*0.333333333f;
	vec3 t_c = (v0 + v1 + v2) * 0.333333333f;
	float beta_v = -0.1666666667f * kv * (totVolume - totVolume0) / (totVolume0);

	f0 += (ksi_3 + cross(t_c, v2-v1)) * beta_v;
	f1 += (ksi_3 + cross(t_c, v0-v2)) * beta_v;
	f2 += (ksi_3 + cross(t_c, v1-v0)) * beta_v;

	int addr0 = warpSize * 3*id0 + mapping[4*i + 0];
	int addr1 = warpSize * 3*id1 + mapping[4*i + 1];
	int addr2 = warpSize * 3*id2 + mapping[4*i + 2];
	//printf("%d\n%d\n%d\n", addr0, addr1, addr2);
	for (int d = 0, sh = 0; d<3; d++, sh+=warpSize)
	{
		fxfyfz[addr0 + sh] = f0[d];
		fxfyfz[addr1 + sh] = f1[d];
		fxfyfz[addr2 + sh] = f2[d];
	}
}


__global__ void perDihedral(int* dihedrals, const float* const xyzuvw, float* fxfyfz, int* mapping, int ndihedrals, int ntriang,
		float kbT, float p, float lmax, float gammaT, float gammaC, float sinTheta0, float cosTheta0, float kb)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ndihedrals) return;

	int id0 = dihedrals[4*i + 0];
	vec3 v0(xyzuvw+6*id0);

	int id1 = dihedrals[4*i + 1];
	vec3 v1(xyzuvw+6*id1);

	int id2 = dihedrals[4*i + 2];
	vec3 v2(xyzuvw+6*id2);

	int id3 = dihedrals[4*i + 3];
	vec3 v3(xyzuvw+6*id3);

	vec3 f0, f1, f2, f3;

	vec3 d21 = v2 - v1;
	float r = norm(d21);
	if (r < 0.0001) r = 0.0001;
	float xx = r/lmax;

	float IbforceI = kbT / p * ( 0.25f/((1.0f-xx)*(1.0f-xx)) - 0.25f + xx ) / r;  // TODO: minus??
	vec3 bforce = d21*IbforceI;
	f1 += bforce;
	f2 -= bforce;

	// Friction force
	vec3 u1(xyzuvw+6*id1 + 3);
	vec3 u2(xyzuvw+6*id2 + 3);
	vec3 du21 = u2 - u1;

	vec3 dforce = du21*gammaT + d21 * gammaC * dot(du21, d21) / (r*r);
	f1 += dforce;
	f2 -= dforce;
	//printf("%f  %f  %f\n", dforce.x, dforce.y, dforce.z);

	vec3 ksi   = cross(v0 - v1, v0 - v2);
	vec3 dzeta = cross(v2 - v3, v1 - v3);
	vec3 t_c0 = (v0 + v1 + v2) * 0.3333333333f;
	vec3 t_c1 = (v1 + v2 + v3) * 0.3333333333f;

	float IksiI = norm(ksi);
	float IdzetaI = norm(dzeta);
	float cosTheta = dot(ksi, dzeta) / (IksiI * IdzetaI);

	float IsinThetaI = sqrt(fabs(1.0f - cosTheta*cosTheta));             // TODO use copysign
	if (fabs(IsinThetaI) < 0.001f) IsinThetaI = 0.001f;

	float sinTheta = IsinThetaI;
	if (dot(ksi - dzeta, t_c0 - t_c1) > 0.0f) sinTheta = -sinTheta;  // ">" because the normals look inside

	float beta_b = kb * (sinTheta * cosTheta0 - cosTheta * sinTheta0) / sinTheta;
	float b11 = -beta_b * cosTheta / (IksiI*IksiI);
	float b12 = beta_b / (IksiI*IdzetaI);
	float b22 = -beta_b * cosTheta / (IdzetaI*IdzetaI);

	f0 += cross(ksi, v2 - v1)*b11 + cross(dzeta, v2 - v1)*b12;
	f1 += cross(ksi, v0 - v2)*b11 + ( cross(ksi, v2 - v3) + cross(dzeta, v0 - v2) )*b12 + cross(dzeta, v2 - v3)*b22;
	f2 += cross(ksi, v1 - v0)*b11 + ( cross(ksi, v3 - v1) + cross(dzeta, v1 - v0) )*b12 + cross(dzeta, v3 - v1)*b22;
	f3 += cross(ksi, v1 - v2)*b12 + cross(dzeta, v1 - v2)*b22;

	int addr0 = warpSize * 3*id0 + mapping[4*(i+ntriang) + 0];
	int addr1 = warpSize * 3*id1 + mapping[4*(i+ntriang) + 1];
	int addr2 = warpSize * 3*id2 + mapping[4*(i+ntriang) + 2];
	int addr3 = warpSize * 3*id3 + mapping[4*(i+ntriang) + 3];
	//printf("%d\n%d\n%d\n%d\n", addr0, addr1, addr2, addr3);
	for (int d = 0, sh = 0; d<3; d++, sh+=warpSize)
	{
		fxfyfz[addr0 + sh] = f0[d];
		fxfyfz[addr1 + sh] = f1[d];
		fxfyfz[addr2 + sh] = f2[d];
		fxfyfz[addr3 + sh] = f3[d];
	}
}

__global__ void collectForces(float* pfxfyfz,  float* fxfyfz, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= warpSize * 3*n) return;

	//if (i< 10) printf("%d\t%f\t%f\t%f\n", i, pfxfyfz[3*i+0], pfxfyfz[3*i+1], pfxfyfz[3*i+2]);

	float myf = pfxfyfz[i];
	myf = warpReduceSum(myf);

	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		fxfyfz[(i / warpSize)] += myf;
	}
	pfxfyfz[i]=0;
}

void forces_nohost(cudaStream_t stream, const float * const device_xyzuvw, float * const device_axayaz)
{
	const int warpSize = 32;

	areaAndVolume(device_xyzuvw);

	//	float *temp = new float[2];
	//	gpuErrchk( cudaMemcpy(temp, totA_V, 2 * sizeof(float), cudaMemcpyDeviceToHost) );
	//	printf("Area:  %.4f,  volume  %.4f\n", temp[0], temp[1]);

	float* pfxfyfz;
	if (bufmap.find(stream) != bufmap.end())
	{
		pfxfyfz = bufmap[stream];
	}
	else
	{
		gpuErrchk( cudaMalloc(&pfxfyfz, 3*warpSize*nparticles * sizeof(float)) );
		bufmap[stream] = pfxfyfz;
	}

	const int threads = 128;
	int blocks  = (ntriang + threads - 1) / threads;
	// Per-triangle forces
	perTriangle<<<blocks, threads, 0, stream>>>(triangles, device_xyzuvw, pfxfyfz, mapping, ntriang,
			params.q, params.Cq, totA_V,
			params.totArea0, params.totVolume0, params.area0,
			params.ka, params.kd, params.kv);

	// Bending force
	blocks  = (ndihedrals + threads - 1) / threads;
	perDihedral<<<blocks, threads, 0, stream>>>(dihedrals, device_xyzuvw, pfxfyfz, mapping, ndihedrals, ntriang,
			params.kbT, params.p, params.lmax,
			params.gammaT, params.gammaC, params.sinTheta0, params.cosTheta0, params.kb);

	// Collect partial sums
	blocks  = (warpSize * 3*nparticles + threads - 1) / threads;
	collectForces<<<blocks, threads, 0, stream>>>(pfxfyfz, device_axayaz, nparticles);
}

void get_triangle_indexing(int (*&host_triplets_ptr)[3], int& ntriangles)
{
	host_triplets_ptr = (int(*)[3])triangles_host;
	ntriangles = CudaRBC::ntriang;
}

}
