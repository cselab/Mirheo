/*
 *  rbcvector.cpp
 *  ctc phenix
 *
 *  Created by Dmitry Alexeev on Nov 17, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */

#include <cassert>
#include <algorithm>

#include "rbcvector.h"
#include "rbcvector-cpu-utils.h"
#include "rbcvector-cuda-utils.h"
#include "cuda-common.h"


void cuda_loadHeader(RBCVector& rbcs, const char* fname, bool report)
{
	cpu_loadHeader(rbcs, fname, report);

	int *devTriangles, *devDihedrals, *devMapping;
	real* devOrig_xyzvuw;
	gpuErrchk( cudaMalloc(&devOrig_xyzvuw, rbcs.nparticles                  * 6 * sizeof(int)) );
	gpuErrchk( cudaMalloc(&devTriangles,   rbcs.ntriang                     * 3 * sizeof(int)) );
	gpuErrchk( cudaMalloc(&devDihedrals,   rbcs.ndihedrals                  * 4 * sizeof(int)) );
	gpuErrchk( cudaMalloc(&devMapping,     (rbcs.ndihedrals + rbcs.ntriang) * 4 * sizeof(int)) );

	gpuErrchk( cudaMemcpy(devOrig_xyzvuw, rbcs.orig_xyzuvw, rbcs.nparticles                  * 6 * sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(devTriangles,   rbcs.triangles,   rbcs.ntriang                     * 3 * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(devDihedrals,   rbcs.dihedrals,   rbcs.ndihedrals                  * 4 * sizeof(int), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(devMapping,     rbcs.mapping,     (rbcs.ndihedrals + rbcs.ntriang) * 4 * sizeof(int), cudaMemcpyHostToDevice) );

	delete[] rbcs.orig_xyzuvw;
	delete[] rbcs.triangles;
	delete[] rbcs.dihedrals;
	delete[] rbcs.mapping;

	rbcs.orig_xyzuvw = devOrig_xyzvuw;
	rbcs.triangles   = devTriangles;
	rbcs.dihedrals   = devDihedrals;
	rbcs.mapping     = devMapping;
}

__global__ void shiftCoos(float* xyzuvw, float dx, float dy, float dz, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	xyzuvw[6*i + 0] += dx;
	xyzuvw[6*i + 1] += dy;
	xyzuvw[6*i + 2] += dz;
}

void cuda_initUnique(RBCVector& rbcs, vector<vector<float> > origins, vector<float> coolo, vector<float> coohi)
{
	// Assume that header is already there

	const int threads = 128;
	const int blocks  = (rbcs.nparticles + threads - 1) / threads;

	// How many cells does my rank have?

	printf("here\n");
	int mine = 0;
	for (int m = 0; m<origins.size(); m++)
	{
		bool inside = true;
		for (int d=0; d<3; d++)
			inside = inside && (coolo[d] < origins[m][d] && origins[m][d] <= coohi[d]);

		if (inside)
			mine++;
	}

	rbcs.n = mine;
	int cellSize = 6 * rbcs.nparticles;
	gpuErrchk( cudaMalloc(&rbcs.xyzuvw, cellSize * mine * sizeof(float)) );
	gpuErrchk( cudaMalloc(&rbcs.ids,    mine * sizeof(int)) );


	// Now copy the coordinates and shift them accordingly

	int cur = 0;
	for (int m = 0; m<origins.size(); m++)
	{
		bool inside = true;
		for (int d=0; d<3; d++)
			inside = inside && (coolo[d] < origins[m][d] && origins[m][d] <= coohi[d]);

		if (inside)
		{
			float* curStart = rbcs.xyzuvw + cur*cellSize;
			gpuErrchk( cudaMemcpy(curStart, rbcs.orig_xyzuvw, cellSize * sizeof(float), cudaMemcpyDeviceToDevice) );
			shiftCoos<<<blocks, threads>>>(curStart,
					origins[m][0] - coolo[0] - 0.5*(coohi[0] - coolo[0]),
					origins[m][1] - coolo[1] - 0.5*(coohi[1] - coolo[1]),
					origins[m][2] - coolo[2] - 0.5*(coohi[2] - coolo[2]),
					rbcs.nparticles);
			cur++;
		}
	}

	gpuErrchk( cudaMalloc(&rbcs.bounds, rbcs.n * 6 * sizeof(float)) );
	gpuErrchk( cudaMalloc(&rbcs.coms,   rbcs.n * 3 * sizeof(float)) );

	// Beware. 28 = 27 + 1, so here i assume no more that 27 ranks
	//  have a cell simultaneously, and 1 extra int (-1) to show
	//  where owners list finishes
	gpuErrchk( cudaMalloc(&rbcs.owners,  rbcs.n * 28 * sizeof(int)) );
	gpuErrchk( cudaMalloc(&rbcs.pfxfyfz, rbcs.n * 3 * 32 * rbcs.nparticles * sizeof(float)) );
}

__inline__ __host__ __device__ float3 fminf(float3 a, float3 b)
{
	return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

__inline__ __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
	return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

__global__ void boundsAndComsKernel(float* xyzuvw, float* globBound, float* globCom, int npart)
{
	float3 loBound = make_float3( 1e10f,  1e10f,  1e10f);
	float3 hiBound = make_float3(-1e10f, -1e10f, -1e10f);
	float3 com     = make_float3(  0.0f,   0.0f,   0.0f);

	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < npart; i += blockDim.x * gridDim.x)
	{
		float3 v = make_float3(xyzuvw[6*i + 0], xyzuvw[6*i + 1], xyzuvw[6*i + 2]);

		loBound = fminf(loBound, v);
		hiBound = fmaxf(hiBound, v);
		com.x   = com.x + v.x;
		com.y   = com.y + v.y;
		com.z   = com.z + v.z;
	}

	loBound = warpReduceMin(loBound);
	__syncthreads();
	hiBound = warpReduceMax(hiBound);
	__syncthreads();
	com = warpReduceSum(com);

	float invN = 1.0f/npart;
	com.x = com.x * invN;
	com.y = com.y * invN;
	com.z = com.z * invN;

	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		atomicAdd(&globCom[0], com.x);
		atomicAdd(&globCom[1], com.y);
		atomicAdd(&globCom[2], com.z);

		atomicAdd(&globBound[0], loBound.x);
		atomicAdd(&globBound[1], loBound.y);
		atomicAdd(&globBound[2], loBound.z);

		atomicAdd(&globBound[3], hiBound.x);
		atomicAdd(&globBound[4], hiBound.y);
		atomicAdd(&globBound[5], hiBound.z);
	}
}

void cuda_boundsAndComs(RBCVector& rbcs)
{
	gpuErrchk( cudaMemset(rbcs.bounds, 0, rbcs.n * 6 * sizeof(float)) );
	gpuErrchk( cudaMemset(rbcs.coms,   0, rbcs.n * 3 * sizeof(float)) );

	const int threads = 128;
	const int blocks  = (rbcs.nparticles + threads - 1) / threads;

	for (int i=0; i<rbcs.n; i++)
	{
		boundsAndComsKernel<<<blocks, threads>>>( rbcs.xyzuvw + 6*i*rbcs.nparticles,
				rbcs.bounds + 6*i, rbcs.coms + 3*i, rbcs.nparticles);
	}
}

