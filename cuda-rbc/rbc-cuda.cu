/*
 *  rbc-cuda.cu
 *  ctc local
 *
 *  Created by Dmitry Alexeev on Nov 3, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */


#include "rbc-cuda.h"

#include <math_functions.h>
#include <cstdio>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <cassert>
#include <cuda_runtime.h>

#include "vec3.h"
#include "cuda-common.h"

using namespace std;

namespace CudaRBC
{

	int nparticles;
	int ntriang;
	int nbonds;
	int ndihedrals;

	int *triangles;
	int *dihedrals;
	int *triangles_host;
	int *triplets;

	// Helper pointers
	int maxCells;
	__constant__ real *totA_V;
	real *host_av;

	// Original configuration
	real* orig_xyzuvw;

	map<cudaStream_t, float*> bufmap;
	__constant__ float A[4][4];

	Extent* dummy;

	texture<float,  cudaTextureType1D> texParticles;
	texture<int4,   cudaTextureType1D> texTriangles4;
	texture<int4,   cudaTextureType1D> texDihedrals4;

    void unitsSetup(float lmax, float p, float cq, float kb, float ka, float kv, float gammaC,
		    float totArea0, float totVolume0, float lunit, float tunit, int ndens, bool prn);
    
	void setup(int& nvertices, Extent& host_extent)
	{
	    const float scale=1;
		
		const bool report = false;

		//        0.0945, 0.00141, 1.642599,
		//        1, 1.8, a, v, a/m.ntriang, 945, 0, 472.5,
		//        90, 30, sin(phi), cos(phi), 6.048

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

			// Scale in dpd units
			xyzuvw_host[6*cur+0] *= scale;
			xyzuvw_host[6*cur+1] *= scale;
			xyzuvw_host[6*cur+2] *= scale;

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

		triangles_host = new int[4*ntriang];
		triplets = new int[3*ntriang];
		for (int i=0; i<ntriang; i++)
		{
			in >> tmp1 >> tmp2 >> id0 >> id1 >> id2;

			id0--; id1--; id2--;
			triangles_host[4*i + 0] = triplets[3*i + 0] = id0;
			triangles_host[4*i + 1] = triplets[3*i + 1] = id1;
			triangles_host[4*i + 2] = triplets[3*i + 2] = id2;
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
		}

		in.close();

		gpuErrchk( cudaMalloc(&orig_xyzuvw, nparticles * 6 * sizeof(float)) );
		gpuErrchk( cudaMalloc(&triangles,   ntriang    * 4 * sizeof(int)) );
		gpuErrchk( cudaMalloc(&dihedrals,   ndihedrals * 4 * sizeof(int)) );

		gpuErrchk( cudaMemcpy(orig_xyzuvw, xyzuvw_host,    nparticles * 6 * sizeof(float), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(triangles,   triangles_host, ntriang    * 4 * sizeof(int),   cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(dihedrals,   dihedrals_host, ndihedrals * 4 * sizeof(int),   cudaMemcpyHostToDevice) );

		delete[] xyzuvw_host;
		delete[] dihedrals_host;

		nvertices = nparticles;
		host_extent.xmin = xmin[0] - origin[0];
		host_extent.ymin = xmin[1] - origin[1];
		host_extent.zmin = xmin[2] - origin[2];

		host_extent.xmax = xmax[0] - origin[0];
		host_extent.ymax = xmax[1] - origin[1];
		host_extent.zmax = xmax[2] - origin[2];

		maxCells = 5;
		gpuErrchk( cudaMalloc(&host_av, maxCells * 2 * sizeof(float)) );
		gpuErrchk( cudaMemcpyToSymbol(totA_V, &host_av,  sizeof(real*)) );

		// Texture setup
		texTriangles4.channelDesc = cudaCreateChannelDesc<int4>();
		texTriangles4.filterMode = cudaFilterModePoint;
		texTriangles4.mipmapFilterMode = cudaFilterModePoint;
		texTriangles4.normalized = 0;

		texDihedrals4.channelDesc = cudaCreateChannelDesc<int4>();
		texDihedrals4.filterMode = cudaFilterModePoint;
		texDihedrals4.mipmapFilterMode = cudaFilterModePoint;
		texDihedrals4.normalized = 0;

		texParticles.channelDesc = cudaCreateChannelDesc<float>();
		texParticles.filterMode = cudaFilterModePoint;
		texParticles.mipmapFilterMode = cudaFilterModePoint;
		texParticles.normalized = 0;

		size_t textureoffset;
		gpuErrchk( cudaBindTexture(&textureoffset, &texTriangles4, triangles, &texTriangles4.channelDesc, ntriang * 4 * sizeof(int)) );
		assert(textureoffset == 0);
		gpuErrchk( cudaBindTexture(&textureoffset, &texDihedrals4, dihedrals, &texDihedrals4.channelDesc, ndihedrals * 4 * sizeof(int)) );
		assert(textureoffset == 0);

		dummy = new Extent[maxCells];
		unitsSetup(1.64, 0.001412*2, 19.0476*0.5, 160, 11004.168, 10159.0438, 2, 135, 91, 1e-6, 2.4295e-6, 4, report);
		//unitsSetup(1.64, 0.00141, 19.0476, 64, 1104.168, 159.0438, 0, 135, 94, 1e-6, 2.4295e-6, 4, report); //unitsSetup(1.64, 0.00705, 6, 15, 1000, 5000, 5, 135, 90, 1e-6, 1e-5, 4, report);
	}

	void unitsSetup(float lmax, float p, float cq, float kb, float ka, float kv, float gammaC,
			float totArea0, float totVolume0, float lunit, float tunit, int ndens, bool prn)
	{
		const float lrbc = 1.000000e-06;
		const float trbc = 3.009441e-03;
		//const float mrbc = 3.811958e-13;

		float ll = lunit / lrbc;
		float tt = tunit / trbc;

		float l0 = 0.537104 / ll;

		params.kbT = 580 * 250 * pow(ll, -2.0) * pow(tt, 2.0);
		params.p = p / ll;
		params.lmax = lmax / ll;
		params.q = 1;
		params.Cq = cq * params.kbT * pow(ll, -2.0);
		params.totArea0 = totArea0 * pow(ll, -2.0);
		params.area0 = params.totArea0 / (float)ntriang;
		params.totVolume0 = totVolume0 * pow(ll, -3.0);
		params.ka =  params.kbT * ka / (l0*l0);
		params.kd =  params.kbT * 0.0 / (l0*l0);
		params.kv =  params.kbT * kv / (l0*l0*l0);//	params.kv =  params.kbT * kv * (l0*l0*l0);
		params.gammaC = gammaC * 580 * pow(tt, 1.0);
		params.gammaT = 3.0 * params.gammaC;

		params.rc = 0.5;
		params.aij = 100;
		params.gamma = 15;
		params.sigma = sqrt(2 * params.gamma * params.kbT);
		//		params.dt = dt;

		float phi = 6.9 / 180.0*M_PI; //float phi = 3.1 / 180.0*M_PI;
		params.sinTheta0 = sin(phi);
		params.cosTheta0 = cos(phi);
		params.kb = kb * params.kbT;

		params.mass = 1.1 / 0.995 * params.totVolume0 * ndens / nparticles;

		params.ndihedrals = ndihedrals;
		params.ntriang = ntriang;
		params.nparticles = nparticles;
		gpuErrchk( cudaMemcpyToSymbol  (devParams, &params, sizeof(Params)) );

		if (prn)
		{
			printf("\n************* Parameters setup *************\n");
			printf("Started with <RBC space (DPD space)>:\n");
			printf("    DPD unit of time:  %e\n",   tunit);
			printf("    DPD unit of length:  %e\n\n", lunit);
			printf("\t Lmax    %12.5f  (%12.5f)\n", lmax,   params.lmax);
			printf("\t p       %12.5f  (%12.5f)\n", p,      params.p);
			printf("\t Cq      %12.5f  (%12.5f)\n", cq,     params.Cq);
			printf("\t kb      %12.5f  (%12.5f)\n", kb,     params.kb);
			printf("\t ka      %12.5f  (%12.5f)\n", ka,     params.ka);
			printf("\t kv      %12.5f  (%12.5f)\n", kv,     params.kv);
			printf("\t gammaC  %12.5f  (%12.5f)\n\n", gammaC, params.gammaC);

			printf("\t kbT     %12e in dpd\n", params.kbT);
			printf("\t mass    %12.5f in dpd\n", params.mass);
			printf("\t area    %12.5f  (%12.5f)\n", totArea0,  params.totArea0);
			printf("\t volume  %12.5f  (%12.5f)\n", totVolume0, params.totVolume0);
			printf("************* **************** *************\n\n");
		}
	}

	int get_nvertices()
	{
		return nparticles;
	}

	Params& get_params()
	{
		return params;
	}

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

	__device__ __inline__ float atomicMin(float *addr, float value)
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

	__device__ __inline__ float atomicMax(float *addr, float value)
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


	__global__ void extentKernel(const float* const __restrict__ xyzuvw, Extent* extent, int npart)
	{
		float3 loBound = make_float3( 1e10f,  1e10f,  1e10f);
		float3 hiBound = make_float3(-1e10f, -1e10f, -1e10f);
		const int cid = blockIdx.y;

		for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < npart; i += blockDim.x * gridDim.x)
		{
			const float* addr = xyzuvw + 6 * (devParams.nparticles*cid + i);
			float3 v = make_float3(addr[0], addr[1], addr[2]);

			loBound = fminf(loBound, v);
			hiBound = fmaxf(hiBound, v);
		}

		loBound = warpReduceMin(loBound);
		__syncthreads();
		hiBound = warpReduceMax(hiBound);

		if ((threadIdx.x & (warpSize - 1)) == 0)
		{
			atomicMin(&extent[cid].xmin, loBound.x);
			atomicMin(&extent[cid].ymin, loBound.y);
			atomicMin(&extent[cid].zmin, loBound.z);

			atomicMax(&extent[cid].xmax, hiBound.x);
			atomicMax(&extent[cid].ymax, hiBound.y);
			atomicMax(&extent[cid].zmax, hiBound.z);
		}
	}

	void extent_nohost(cudaStream_t stream, int ncells, const float * const xyzuvw, Extent * device_extent, int n)
	{
		if (ncells == 0) return;

		dim3 threads(32*3, 1);
		dim3 blocks( (ntriang + threads.x - 1) / threads.x, ncells );

		if (ncells > maxCells)
		{
			maxCells = 2*ncells;
			gpuErrchk( cudaFree(host_av) );
			gpuErrchk( cudaMalloc(&host_av, maxCells * 2 * sizeof(float)) );
			gpuErrchk( cudaMemcpyToSymbol(totA_V, &host_av,  sizeof(real*)) );

			delete[] dummy;
			dummy = new Extent[maxCells];
		}

		for (int i=0; i<ncells; i++)
		{
			dummy[i].xmin = dummy[i].ymin = dummy[i].zmin = 1e10;
			dummy[i].xmax = dummy[i].ymax = dummy[i].zmax = -1e10;
		}

		gpuErrchk( cudaMemcpy(device_extent, dummy, ncells * sizeof(Extent), cudaMemcpyHostToDevice) );

		if (n == -1) n = nparticles;
		extentKernel<<<blocks, threads, 0, stream>>>(xyzuvw, device_extent, n);
		gpuErrchk( cudaPeekAtLastError() );
	}

	__device__ __inline__ vec3 tex2vec(int id)
	{
		return vec3(tex1Dfetch(texParticles, id+0),
				tex1Dfetch(texParticles, id+1),
				tex1Dfetch(texParticles, id+2));
	}

	__global__ void areaAndVolumeKernel()
	{
		float2 a_v = make_float2(0.0f, 0.0f);
		const int cid = blockIdx.y;

		for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < devParams.ntriang; i += blockDim.x * gridDim.x)
		{
			int4 ids = tex1Dfetch(texTriangles4, i);
			vec3 v0( tex2vec(6*(ids.x+cid*devParams.nparticles)) );
			vec3 v1( tex2vec(6*(ids.y+cid*devParams.nparticles)) );
			vec3 v2( tex2vec(6*(ids.z+cid*devParams.nparticles)) );

			a_v.x += 0.5f * norm(cross(v1 - v0, v2 - v0));
			a_v.y += 0.1666666667f * (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
					- v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
		}

		a_v = warpReduceSum(a_v);
		if ((threadIdx.x & (warpSize - 1)) == 0)
		{
			atomicAdd(&totA_V[2*cid+0], a_v.x);
			atomicAdd(&totA_V[2*cid+1], a_v.y);
		}
	}

	__global__ void perTriangle(float* fxfyfz)
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;
		const int cid = blockIdx.y;
		if (i >= devParams.ntriang) return;

		const float totArea   =      totA_V[2*cid + 0];
		const float totVolume = fabs(totA_V[2*cid + 1]);

		int4 ids = tex1Dfetch(texTriangles4, i);
		vec3 v0( tex2vec(6*(ids.x+cid*devParams.nparticles)) );
		vec3 v1( tex2vec(6*(ids.y+cid*devParams.nparticles)) );
		vec3 v2( tex2vec(6*(ids.z+cid*devParams.nparticles)) );

		vec3 ksi = cross(v1 - v0, v2 - v0);
		float area = 0.5f * norm(ksi);

		// in-plane
		float alpha = 0.25f * devParams.q*devParams.Cq / powf(area, devParams.q+2.0f);

		// area conservation
		float beta_a = -0.25f * ( devParams.ka*(totArea - devParams.totArea0) / (devParams.totArea0*area) +
				devParams.kd * (area - devParams.area0) / (devParams.area0 * area) );
		alpha += beta_a;
		vec3 f0, f1, f2;

		f0 = cross(ksi, v2-v1)*alpha;
		f1 = cross(ksi, v0-v2)*alpha;
		f2 = cross(ksi, v1-v0)*alpha;

		// volume conservation
		// "-" here is because the normals look inside
		vec3 ksi_3 = ksi*0.333333333f;
		vec3 t_c = (v0 + v1 + v2) * 0.333333333f;
		float beta_v = -0.1666666667f * devParams.kv * (totVolume - devParams.totVolume0) / (devParams.totVolume0);

		f0 += (ksi_3 + cross(t_c, v2-v1)) * beta_v;
		f1 += (ksi_3 + cross(t_c, v0-v2)) * beta_v;
		f2 += (ksi_3 + cross(t_c, v1-v0)) * beta_v;

		float* addr = fxfyfz + 3*cid*devParams.nparticles;
#pragma unroll
		for (int d = 0; d<3; d++)
		{
			atomicAdd(addr + 3*ids.x + d, f0[d]);
			atomicAdd(addr + 3*ids.y + d, f1[d]);
			atomicAdd(addr + 3*ids.z + d, f2[d]);
		}
	}

	__global__ void perDihedral(float* fxfyfz)
	{
		const int i = blockIdx.x * blockDim.x + threadIdx.x;
		const int cid = blockIdx.y;
		if (i >= devParams.ndihedrals) return;

		int4 ids = tex1Dfetch(texDihedrals4, i);
		vec3 v0( tex2vec(6*(ids.x+cid*devParams.nparticles)) );
		vec3 v1( tex2vec(6*(ids.y+cid*devParams.nparticles)) );
		vec3 v2( tex2vec(6*(ids.z+cid*devParams.nparticles)) );
		vec3 v3( tex2vec(6*(ids.w+cid*devParams.nparticles)) );

		vec3 f0, f1, f2, f3;

		vec3 d21 = v2 - v1;
		float r = norm(d21);
		if (r < 0.0001) r = 0.0001;
		float xx = r/devParams.lmax;

		float IbforceI = devParams.kbT / devParams.p * ( 0.25f/((1.0f-xx)*(1.0f-xx)) - 0.25f + xx ) / r;  // TODO: minus??
		vec3 bforce = d21*IbforceI;
		f1 += bforce;
		f2 -= bforce;

		// Friction force
		vec3 u1( tex2vec(6*(ids.y+cid*devParams.nparticles) + 3) );
		vec3 u2( tex2vec(6*(ids.z+cid*devParams.nparticles) + 3) );

		vec3 du21 = u2 - u1;

		vec3 dforce = du21*devParams.gammaT + d21 * devParams.gammaC * dot(du21, d21) / (r*r);
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

		float beta_b = devParams.kb * (sinTheta * devParams.cosTheta0 - cosTheta * devParams.sinTheta0) / sinTheta;
		float b11 = -beta_b * cosTheta / (IksiI*IksiI);
		float b12 = beta_b / (IksiI*IdzetaI);
		float b22 = -beta_b * cosTheta / (IdzetaI*IdzetaI);

		f0 += cross(ksi, v2 - v1)*b11 + cross(dzeta, v2 - v1)*b12;
		f1 += cross(ksi, v0 - v2)*b11 + ( cross(ksi, v2 - v3) + cross(dzeta, v0 - v2) )*b12 + cross(dzeta, v2 - v3)*b22;
		f2 += cross(ksi, v1 - v0)*b11 + ( cross(ksi, v3 - v1) + cross(dzeta, v1 - v0) )*b12 + cross(dzeta, v3 - v1)*b22;
		f3 += cross(ksi, v1 - v2)*b12 + cross(dzeta, v1 - v2)*b22;

		float* addr = fxfyfz + 3*cid*devParams.nparticles;
#pragma unroll
		for (int d = 0; d<3; d++)
		{
			atomicAdd(addr + 3*ids.x + d, f0[d]);
			atomicAdd(addr + 3*ids.y + d, f1[d]);
			atomicAdd(addr + 3*ids.z + d, f2[d]);
			atomicAdd(addr + 3*ids.w + d, f3[d]);
		}
	}

	void forces_nohost(cudaStream_t stream, int ncells, const float * const device_xyzuvw, float * const device_axayaz)
	{
		if (ncells == 0) return;

		if (ncells > maxCells)
		{
			maxCells = 2*ncells;
			gpuErrchk( cudaFree(host_av) );
			gpuErrchk( cudaMalloc(&host_av, maxCells * 2 * sizeof(float)) );
			gpuErrchk( cudaMemcpyToSymbol(totA_V, &host_av,  sizeof(real*)) );

			delete[] dummy;
			dummy = new Extent[maxCells];
		}

		size_t textureoffset;
		gpuErrchk( cudaBindTexture(&textureoffset, &texParticles,  device_xyzuvw, &texParticles.channelDesc,  ncells * nparticles * 6 * sizeof(float)) );
		assert(textureoffset == 0);

		dim3 trThreads(32*3, 1);
		dim3 trBlocks( (ntriang + trThreads.x - 1) / trThreads.x, ncells );

		dim3 dihThreads(32*3, 1);
		dim3 dihBlocks( (ndihedrals + dihThreads.x - 1) / dihThreads.x, ncells );

		gpuErrchk( cudaMemset(host_av, 0, ncells * 2 * sizeof(float)) );
		areaAndVolumeKernel<<<trBlocks, trThreads, 0, stream>>>();
		gpuErrchk( cudaPeekAtLastError() );

//		float *temp = new float[ncells*2];
//		gpuErrchk( cudaMemcpy(temp, host_av, ncells * 2 * sizeof(float), cudaMemcpyDeviceToHost) );
//		for (int i=0; i<ncells; i++)
//			printf("# %d:  Area:  %.4f,  volume  %.4f\n", i, temp[2*i], temp[2*i+1]);

		perDihedral<<<dihBlocks, dihThreads, 0, stream>>>(device_axayaz);
		gpuErrchk( cudaPeekAtLastError() );
		perTriangle<<<trBlocks, trThreads, 0, stream>>>(device_axayaz);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaUnbindTexture(texParticles) );
	}

	void get_triangle_indexing(int (*&host_triplets_ptr)[3], int& ntriangles)
	{
		host_triplets_ptr = (int(*)[3])triplets;
		ntriangles = ntriang;
	}

	float* get_orig_xyzuvw()
	{
		return orig_xyzuvw;
	}

}
