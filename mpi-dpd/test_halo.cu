
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cassert>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include "../cuda-dpd/cell-lists.h"
#include <helper_math.h>

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

		abort();
	}
}

struct Particle
{
	float x[3], u[3];
};

struct Acceleration
{
	float a[3];
};

//container for the gpu particles during the simulation
template<typename T>
struct SimpleDeviceBuffer
{
	int capacity, size;
	T * devdata;

	SimpleDeviceBuffer(int n = 0): capacity(0), size(0), devdata(NULL) { resize(n); }

	~SimpleDeviceBuffer()
	{
		if (devdata != NULL) CUDA_CHECK(cudaFree(devdata));
	}

	void resize(const int n)
	{
		assert(n >= 0);
		size = n;

		if (capacity >= n) return;

		if (devdata != NULL) CUDA_CHECK(cudaFree(devdata));

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_CHECK(cudaMalloc(&devdata, sizeof(T) * capacity));
	}

	void preserve_resize(const int n)
	{
		assert(n >= 0);

		T * old = devdata;

		const int oldsize = size;
		size = n;

		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_CHECK( cudaMalloc(&devdata, sizeof(T) * capacity) );

		if (old != NULL)
		{
			CUDA_CHECK(cudaMemcpy(devdata, old, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaFree(old));
		}
	}

	template<typename Cont>
	void copy(Cont& cont)
	{
		static_assert(std::is_same<decltype(devdata), decltype(cont.devdata)>::value, "can't copy buffers of different types");

		resize(cont.size);
		CUDA_CHECK( cudaMemcpy(devdata, cont.devdata, sizeof(T) * size, cudaMemcpyDeviceToDevice) );
	}

};

template<typename T>
struct PinnedHostBuffer
{
	int capacity, size;
	T * hostdata, * devdata;

	PinnedHostBuffer(int n = 0): capacity(0), size(0), hostdata(NULL), devdata(NULL) { resize(n); }

	~PinnedHostBuffer()
	{
		if (hostdata != NULL) CUDA_CHECK(cudaFreeHost(hostdata));
	}

	void resize(const int n)
	{
		assert(n >= 0);

		size = n;
		if (capacity >= n) return;

		if (hostdata != NULL) CUDA_CHECK(cudaFreeHost(hostdata));

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		CUDA_CHECK(cudaHostAlloc(&hostdata, sizeof(T) * capacity, cudaHostAllocMapped));
		CUDA_CHECK(cudaHostGetDevicePointer(&devdata, hostdata, 0));
	}

	void preserve_resize(const int n)
	{
		assert(n >= 0);

		T * old = hostdata;

		const int oldsize = size;
		size = n;

		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		hostdata = NULL;
		CUDA_CHECK(cudaHostAlloc(&hostdata, sizeof(T) * capacity, cudaHostAllocMapped));

		if (old != NULL)
		{
			CUDA_CHECK(cudaMemcpy(hostdata, old, sizeof(T) * oldsize, cudaMemcpyHostToHost));
			CUDA_CHECK(cudaFreeHost(old));
		}

		CUDA_CHECK(cudaHostGetDevicePointer(&devdata, hostdata, 0));
	}

	T& operator[](const int i)
	{
		return hostdata[i];
	}

	template<typename Cont>
	void copy(Cont& cont)
	{
		static_assert(std::is_same<decltype(devdata), decltype(cont.devdata)>::value, "can't copy buffers of different types");

		resize(cont.size);
		CUDA_CHECK( cudaMemcpy(devdata, cont.devdata, sizeof(T) * size, cudaMemcpyDeviceToDevice) );
	}
};


//const int variant = blockIdx.x;
//int cx, cy, cz;
//
//switch (variant)
//{
//	case 0: cx = 0;    cy = blockIdx.y * threadId.x; cz = blockIdx.z; break;
//	case 1: cx = nx-1; cy = blockIdx.y * threadId.x; cz = blockIdx.z; break;
//	case 2: cy = 0;    cz = blockIdx.y; cx = blockIdx.z * threadId.x; break;
//	case 3: cy = ny-1; cz = blockIdx.y; cx = blockIdx.z * threadId.x; break;
//	case 4: cz = 0;    cx = blockIdx.y * threadId.x; cy = blockIdx.z; break;
//	case 5: cz = nz-1; cx = blockIdx.y * threadId.x; cy = blockIdx.z; break;
//}
//
//if (cx >= nx || cy >= ny || cz >= nz) return;
//
//const int hidx = (cx == 0) ? 0 : ((cx == nx-1) ? 2 : 1);
//const int hidy = (cy == 0) ? 0 : ((cy == ny-1) ? 2 : 1);
//const int hidz = (cz == 0) ? 0 : ((cz == nz-1) ? 2 : 1);
//
//const int cid = (cz*ny + cy) * nx + cx;
//if (cid >= ncells || completed[cid] > 0) return;


__launch_bounds__(128, 16)
__global__ void getHalos(const int* __restrict__ cellsstart, const float2* __restrict__ xyzuvw, const int nx, const int ny, const int nz, const int ncells,
		float2* dst0, float2* dst1, float2* dst2, float2* dst3, int count[4], int* limits)
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int variant = blockIdx.y;
	int cid;

	if (variant <= 1)
		cid = gid * nx  +  (nx-1) * variant; // x = 0,   x = nx - 1
	else
		cid = (gid % nx) + (gid / nx) * nx*(ny-1)  +  nx * (ny-1) * (variant - 2);  // y = 0,   y = ny - 1

	if (cid >= ncells) return;

	volatile __shared__ int shIndex;
	__shared__ int bsize;
	bsize = 0;

	const int pstart = cellsstart[cid];
	const int pend   = cellsstart[cid+1];

	int myid = atomicAdd(&bsize, pend-pstart);

	__syncthreads();

	if (threadIdx.x == 0)
		shIndex = atomicAdd(count + variant, bsize);

	__syncthreads();

	myid += shIndex;

	float2* dest[4] = {dst0, dst1, dst2, dst3};

	for (int i = 0; i < pend-pstart; i++)
	{
		const int dstInd = 3*(myid   + i);
		const int srcInd = 3*(pstart + i);

		dest[variant][dstInd + 0] = xyzuvw[srcInd + 0];
		dest[variant][dstInd + 1] = xyzuvw[srcInd + 1];
		dest[variant][dstInd + 2] = xyzuvw[srcInd + 2];
	}

	if (gid == 0)
	{
		limits[0] = cellsstart[nx*ny];
		limits[1] = cellsstart[ncells - nx*ny];
	}
}

int main(int argc, char **argv)
{
	const int nx = 48;
	const int ny = 48;
	const int nz = 48;
	const int ncells = nx*ny*nz;

	const int ndens = 4;
	const int maxdim = std::max({nx, ny, nz});
	const int np = ncells*ndens;
	PinnedHostBuffer<Particle>   particles(np);
	SimpleDeviceBuffer<Particle> devp(np);
	SimpleDeviceBuffer<float4>   pWithO(np*2);
	SimpleDeviceBuffer<ushort4>  halfPwithO(np*2);

	SimpleDeviceBuffer<int> cellsstart(ncells + 1);
	SimpleDeviceBuffer<int> cellscount(ncells);

	PinnedHostBuffer<int> hcellsstart(ncells + 1);

	SimpleDeviceBuffer<Acceleration> acc(np);
	PinnedHostBuffer  <Acceleration> hacc(np);

	const int cap = maxdim*maxdim*ndens*3;
	SimpleDeviceBuffer<Particle> haloBuffers[4] = {cap, cap, cap, cap};

	std::vector<Particle> halos[3][3][3];

	for (int dx = -1; dx <= 1; dx++)
		for (int dy = -1; dy <= 1; dy++)
			for (int dz = -1; dz <= 1; dz++)
			{
				const int sum = std::abs(dx) + std::abs(dy) + std::abs(dz);
				if (sum > 0) halos[dx+1][dy+1][dz+1].resize(3 * ndens * pow(maxdim, 3 - sum));
			}

	srand48(0);

	printf("initializing...\n");

	int c = 0;
	for (int i=0; i<nx; i++)
		for (int j=0; j<ny; j++)
			for (int k=0; k<nz; k++)
				for (int p=0; p<ndens; p++)
				{
					particles.hostdata[c].x[0] = i + drand48() - nx/2;
					particles.hostdata[c].x[1] = j + drand48() - ny/2;
					particles.hostdata[c].x[2] = k + drand48() - nz/2;

					particles.hostdata[c].u[0] = drand48() - 0.5;
					particles.hostdata[c].u[1] = drand48() - 0.5;
					particles.hostdata[c].u[2] = drand48() - 0.5;
					c++;
				}

	printf("building cells...\n");

	build_clists((float*)particles.devdata, np, 1.0, nx, ny, nz, -nx/2.0, -ny/2.0, -nz/2.0, nullptr, cellsstart.devdata, cellscount.devdata);

	CUDA_CHECK( cudaMemcpy(devp.devdata, particles.devdata, np * sizeof(Particle), cudaMemcpyDeviceToDevice) );
	CUDA_CHECK( cudaMemcpy(hcellsstart.devdata, cellsstart.devdata, cellsstart.size * sizeof(int), cudaMemcpyDeviceToDevice) );

	SimpleDeviceBuffer<int> counts(4), limits(2);
	PinnedHostBuffer<int> hcounts, hlimits;

	const int nthreads = 64;
	CUDA_CHECK( cudaMemset(counts.devdata, 0, 4*sizeof(int)) );
	getHalos<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 4, 1),  dim3(nthreads, 1, 1) >>>(cellsstart.devdata, (float2*)devp.devdata, nx, ny, nz, nx*ny*nz,
			(float2*)haloBuffers[0].devdata, (float2*)haloBuffers[1].devdata, (float2*)haloBuffers[2].devdata, (float2*)haloBuffers[3].devdata, counts.devdata, limits.devdata);

	CUDA_CHECK( cudaDeviceSynchronize() );

	hcounts.copy(counts);
	hlimits.copy(limits);
	printf("%d %d %d %d\n", hcounts[0], hcounts[1], hcounts[2], hcounts[3]);

//	CUDA_CHECK( cudaMemcpyAsync(&halos[0][1][1][0], haloBuffers[0].devdata, hcounts[0] * sizeof(Particle), cudaMemcpyDeviceToHost, 0) );
//	CUDA_CHECK( cudaMemcpyAsync(&halos[2][1][1][0], haloBuffers[0].devdata, hcounts[1] * sizeof(Particle), cudaMemcpyDeviceToHost, 0) );
//	CUDA_CHECK( cudaMemcpyAsync(&halos[1][0][1][0], haloBuffers[0].devdata, hcounts[2] * sizeof(Particle), cudaMemcpyDeviceToHost, 0) );
//	CUDA_CHECK( cudaMemcpyAsync(&halos[1][2][1][0], haloBuffers[0].devdata, hcounts[3] * sizeof(Particle), cudaMemcpyDeviceToHost, 0) );
////
//	printf("lims %d %d\n", hlimits[0], hlimits[1]);
//	CUDA_CHECK( cudaMemcpyAsync(&halos[1][1][0][0], (float2*)devp.devdata, hlimits[0] * sizeof(Particle), cudaMemcpyDeviceToHost, 0) );
//	CUDA_CHECK( cudaMemcpyAsync(&halos[1][1][2][0], (float2*)devp.devdata + 3*hlimits[1], (np - hlimits[1]) * sizeof(Particle), cudaMemcpyDeviceToHost, 0) );
//
//	CUDA_CHECK( cudaDeviceSynchronize() );


	return 0;
}
