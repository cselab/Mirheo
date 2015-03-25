/*
 *  cell-lists-faster.cu
 *  Part of CTC/cuda-dpd-sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-08-07.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cassert>

texture<float, cudaTextureType1D> texParticlesCLS;

__device__ int  blockscount = 0;
 
template<int ILP, int SLOTS, int WARPS>
__global__ void yzhistogram(const int np,
			    const float invrc, const int3 ncells, 
			    const float3 domainstart,
			    int * const yzcid,
			    int * const localoffsets,
			    int * const global_yzhisto,
			    int * const global_yzscan,
			    int * const max_yzcount)
{
    extern __shared__ int shmemhisto[];

    assert(blockDim.y == 1);
    assert(blockDim.x == warpSize * WARPS);

    const int tid = threadIdx.x;
    const int slot = tid % (SLOTS);
    const int gsize = gridDim.x * blockDim.x;
    const int nhisto = ncells.y * ncells.z;

    const int tile = blockIdx.x * blockDim.x;
    
    if (tile >= np)
	return;
        
    for(int i = tid ; i < SLOTS * nhisto; i += blockDim.x)
	shmemhisto[i] = 0;
 
    float y[ILP], z[ILP];
    for(int j = 0; j < ILP; ++j)
    {
	const int g = tile + tid + gsize * j;

	y[j] = z[j] = -1;

	if (g < np)
	{
	    y[j] = tex1Dfetch(texParticlesCLS, 1 + 6 * g); 
	    z[j] = tex1Dfetch(texParticlesCLS, 2 + 6 * g); 
	}
    }

    __syncthreads();
	
    int entries[ILP], offset[ILP];
    for(int j = 0; j < ILP; ++j)
    {
	const int g = tile + tid + gsize * j;
	    
	int ycid = min(ncells.y - 1, max(0, (int)(floor(y[j] - domainstart.y) * invrc)));
	int zcid = min(ncells.z - 1, max(0, (int)(floor(z[j] - domainstart.z) * invrc)));
	    
	assert(ycid >= 0 && ycid < ncells.y);
	assert(zcid >= 0 && zcid < ncells.z);

	entries[j] = -1;
	offset[j] = -1;

	if (g < np)
	{
	    entries[j] =  ycid + ncells.y * zcid;
	    offset[j] = atomicAdd(shmemhisto + entries[j] + slot * nhisto, 1);
	}
    }

    __syncthreads();
	
    for(int s = 1; s < SLOTS; ++s)
    {
	for(int i = tid ; i < nhisto; i += blockDim.x)
	    shmemhisto[i + s * nhisto] += shmemhisto[i + (s - 1) * nhisto];

	__syncthreads();
    }

    if (slot > 0)
	for(int j = 0; j < ILP; ++j)
	    offset[j] += shmemhisto[entries[j] + (slot - 1) * nhisto];
	
    __syncthreads();
	
    for(int i = tid ; i < nhisto; i += blockDim.x)
	shmemhisto[i] = atomicAdd(global_yzhisto + i, shmemhisto[i + (SLOTS - 1) * nhisto]);

    __syncthreads();

    for(int j = 0; j < ILP; ++j)
    {
	const int g = tile + tid + gsize * j;
	    
	if (g < np)
	{
	    yzcid[g] = entries[j];
	    localoffsets[g] = offset[j] + shmemhisto[entries[j]];
	}
    }
    
    __shared__ bool lastone;

    if (tid == 0)
    {
	lastone = gridDim.x - 1 == atomicAdd(&blockscount, 1);
	
	if (lastone)
	    blockscount = 0;
    }

    __syncthreads();
        
    if (lastone)
    {
	for(int i = tid ; i < nhisto; i += blockDim.x)
	    shmemhisto[i] = global_yzhisto[i];

	if (max_yzcount != NULL)
	{
	    __syncthreads();

	    int mymax = 0;
	    for(int i = tid ; i < nhisto; i += blockDim.x)
		mymax = max(shmemhisto[i], mymax);

	    for(int L = 16; L > 0; L >>=1)
		mymax = max(__shfl_xor(mymax, L), mymax);

	    __shared__ int maxies[WARPS];
	
	    if (tid % warpSize == 0)
		maxies[tid / warpSize] = mymax;
	
	    __syncthreads();

	    mymax = 0;
	
	    if (tid < WARPS)
		mymax = maxies[tid];

	    for(int L = 16; L > 0; L >>=1)
		mymax = max(__shfl_xor(mymax, L), mymax);

	    if (tid == 0)
		*max_yzcount = mymax;
	}
	
	const int bwork = blockDim.x * ILP;
	for(int tile = 0; tile < nhisto; tile += bwork)
	{
	    const int n = min(bwork, nhisto - tile);

	    __syncthreads();
	    
	    if (tile > 0 && tid == 0)
		shmemhisto[tile] += shmemhisto[tile - 1];
	    
	    for(int l = 1; l < n; l <<= 1)
	    {
		__syncthreads();
		
		int tmp[ILP];

		for(int j = 0; j < ILP; ++j)
		{
		    const int d = tid + j * blockDim.x;
		    
		    tmp[j] = 0;

		    if (d >= l && d < n) 
			tmp[j] = shmemhisto[d + tile] + shmemhisto[d + tile - l];
		}

		__syncthreads();

		for(int j = 0; j < ILP; ++j)
		{
		    const int d = tid + j * blockDim.x;

		    if (d >= l && d < n) 
			shmemhisto[d + tile] = tmp[j];
		}
	    }
	}

	for(int i = tid ; i < nhisto; i += blockDim.x)
	    global_yzscan[i] = i == 0 ? 0 : shmemhisto[i - 1];
    }
}

texture<int, cudaTextureType1D> texScanYZ;

template<int ILP>
__global__ void yzscatter(const int * const localoffsets,
			  const int * const yzcids,
			  const int np,
			  int * const outid)
{
    for(int j = 0; j < ILP; ++j)
    {
	const int g = threadIdx.x + blockDim.x * (j + ILP * blockIdx.x);

	if (g < np)
	{
	    const int yzcid = yzcids[g];
	    const int localoffset = localoffsets[g];
	    const int base = tex1Dfetch(texScanYZ, yzcid);
	
	    const int entry = base + localoffset;

	    outid[entry] = g;
	}
    }
}

texture<int, cudaTextureType1D> texCountYZ;

template<int YCPB>
__global__ void xgather(const int * const ids, const int np, const float invrc, const int3 ncells, const float3 domainstart,
			int * const starts, int * const counts,
			float * const xyzuvw, const int bufsize, int * const order)
{
    assert(gridDim.x == 1 && gridDim.y == ncells.y / YCPB && gridDim.z == ncells.z);
    assert(blockDim.x == warpSize);
    assert(blockDim.y == YCPB);
    
    extern __shared__ volatile int allhisto[];
    volatile int * const xhisto = &allhisto[ncells.x * threadIdx.y];
    volatile int * const loffset = &allhisto[YCPB * ncells.x + bufsize * threadIdx.y];
    volatile int * const reordered = &allhisto[YCPB * ncells.x + bufsize * (YCPB + threadIdx.y)];

    const int tid = threadIdx.x;
    const int ycid = threadIdx.y + YCPB * blockIdx.y;

    if (ycid >= ncells.y)
	return;
    
    const int yzcid = ycid + ncells.y * blockIdx.z;
    const int start = tex1Dfetch(texScanYZ, yzcid);
    const int count = tex1Dfetch(texCountYZ, yzcid);

    if (count > bufsize)
    {
	//asm("trap ;");
	return; //something went wrong
    }
    
    for(int i = tid; i < ncells.x; i += warpSize)
	xhisto[i] = 0;
 
    for(int i = tid; i < count; i += warpSize)
    {
	const int g = start + i;

 	const int id = ids[g];

	const float x = tex1Dfetch(texParticlesCLS, 6 * id);

	const int xcid = min(ncells.x - 1, max(0, (int)floor(invrc * (x - domainstart.x))));
	
	const int val = atomicAdd((int *)(xhisto + xcid), 1);
	assert(xcid < ncells.x);
	assert(i < bufsize);
	
	loffset[i] = val |  (xcid << 16);
    }
    
    for(int i = tid; i < ncells.x; i += warpSize)
	counts[i + ncells.x * yzcid] = xhisto[i];

    for(int base = 0; base < ncells.x; base += warpSize)
    {
	const int n = min(warpSize, ncells.x - base);
	const int g = base + tid;
	
	int val = (tid == 0 && base > 0) ? xhisto[g - 1] : 0;

	if (tid < n)
	    val += xhisto[g];

	for(int l = 1; l < n; l <<= 1)
	    val += (tid >= l) * __shfl_up(val, l);

	if (tid < n)
	    xhisto[g] = val;
    }

    for(int i = tid; i < ncells.x; i += warpSize)
	starts[i + ncells.x * yzcid] = start + (i == 0 ? 0 : xhisto[i - 1]);
 
    for(int i = tid; i < count; i += warpSize)
    {
	const int entry = loffset[i];
	const int xcid = entry >> 16;
	assert(xcid < ncells.x);
	const int loff = entry & 0xffff;

	const int dest = (xcid == 0 ? 0 : xhisto[xcid - 1]) + loff;

	reordered[dest] = ids[start + i];
    }

    const int nfloats = count * 6;
    const int base = 6 * start;
    
    const int mystart = (32 - (base & 0x1f) + tid) & 0x1f;
    for(int i = mystart; i < nfloats; i += warpSize)
    {
	const int c = i % 6;
	const int p = reordered[i / 6];
	assert(i / 6 < bufsize);
	
	xyzuvw[base + i] = tex1Dfetch(texParticlesCLS, c + 6 * p);
    }

    if (order != NULL)
	for(int i = tid; i < count; i += warpSize)
	    order[start + i] = reordered[i];
}

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

using namespace thrust;

#include "hacks.h"

#include <utility>

struct FailureTest
{
    int bufsize;
    int * maxstripe, * dmaxstripe;

    FailureTest() : maxstripe(NULL), dmaxstripe(NULL)
	{
	    cudaDeviceProp prop;
	    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	    if (!prop.canMapHostMemory)
	    {
		printf("Capability zero-copy not there! Aborting now.\n");
		abort();
	    }
	    else
	    { 
		cudaSetDeviceFlags(cudaDeviceMapHost);
		cudaError_t status = cudaGetLastError ( );
		cudaError_t status2 = cudaPeekAtLastError();

		//printf("attempting to set MapHost..status:  %d -> %d\n", status == cudaSuccess, status2 == cudaSuccess);
	    }
	}

    static void callback_crash(cudaStream_t stream, cudaError_t status, void*  userData )
	{
	    FailureTest& f = *(FailureTest *)userData;
	    
	    if (*f.maxstripe > f.bufsize)
	    {
		printf("Ouch .. I would need to rollback. Maxstripe: %d, bufsize: %d\n", *f.maxstripe, f.bufsize);
		printf("Too late to recover this. Aborting now.\n");
		abort();
	    }
	}

    void reset() 
	{
	    if (maxstripe == NULL)
	    {
		CUDA_CHECK(cudaHostAlloc((void **)&maxstripe, sizeof(int), cudaHostAllocMapped));
		assert(maxstripe != NULL);
		
		CUDA_CHECK(cudaHostGetDevicePointer(&dmaxstripe, maxstripe, 0));
	    }
	    
	    *maxstripe = 0; 
	}
} static failuretest;

struct is_gzero
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return  x > 0;
  }
};

bool clists_perfmon = false;
bool clists_robust = true;

float * xyzuvw_copy = NULL;
int *loffsets = NULL, *yzcid = NULL, *outid = NULL, *order= NULL, *dyzscan = NULL, *yzhisto = NULL;

cudaEvent_t evstart, evacquire, evscatter, evgather;

bool initialized = false;
int old_np = 0, old_yzncells = 0;

void build_clists(float * const device_xyzuvw, int np, const float rc, 
		  const int xcells, const int ycells, const int zcells,
		  const float xdomainstart, const float ydomainstart, const float zdomainstart,
		  int * const host_order, int * device_cellsstart, int * device_cellscount,
		  std::pair<int, int *> * nonemptycells, cudaStream_t stream)
{
    assert(np > 0);
    
    const float3 domainstart = make_float3(xdomainstart, ydomainstart, zdomainstart);
    const int3 ncells = make_int3(xcells, ycells, zcells);
    const int yzncells = ycells * zcells;
    const float densitynumber = np / (float)(ncells.x * ncells.y * ncells.z);
    int xbufsize = (int)(ncells.x * densitynumber * 2);
     
    if (!initialized)
    {
	CUDA_CHECK(cudaEventCreate(&evacquire));
	CUDA_CHECK(cudaEventCreate(&evstart));
	CUDA_CHECK(cudaEventCreate(&evscatter));
	CUDA_CHECK(cudaEventCreate(&evgather));
	    
	texScanYZ.channelDesc = cudaCreateChannelDesc<int>();
	texScanYZ.filterMode = cudaFilterModePoint;
	texScanYZ.mipmapFilterMode = cudaFilterModePoint;
	texScanYZ.normalized = 0;
    
	texCountYZ.channelDesc = cudaCreateChannelDesc<int>();
	texCountYZ.filterMode = cudaFilterModePoint;
	texCountYZ.mipmapFilterMode = cudaFilterModePoint;
	texCountYZ.normalized = 0;

	texParticlesCLS.channelDesc = cudaCreateChannelDesc<float>();
	texParticlesCLS.filterMode = cudaFilterModePoint;
	texParticlesCLS.mipmapFilterMode = cudaFilterModePoint;
	texParticlesCLS.normalized = 0;
	
	initialized = true;
    }

    if (old_np < np)
    {
	if (old_np > 0)
	{
	    CUDA_CHECK(cudaFree(xyzuvw_copy));
	    CUDA_CHECK(cudaFree(loffsets));
	    CUDA_CHECK(cudaFree(yzcid));
	    CUDA_CHECK(cudaFree(outid));
	    CUDA_CHECK(cudaFree(order));
	}

	CUDA_CHECK(cudaMalloc(&xyzuvw_copy, sizeof(float) * 6 * np));
	CUDA_CHECK(cudaMalloc(&loffsets, sizeof(int) * np));
	CUDA_CHECK(cudaMalloc(&yzcid, sizeof(int) * np));
	CUDA_CHECK(cudaMalloc(&outid, sizeof(int) * np));
	CUDA_CHECK(cudaMalloc(&order, sizeof(int) * np));

	old_np = np;
    }

    
    if (old_yzncells < yzncells)
    {
	if (old_yzncells > 0)
	{
	    CUDA_CHECK(cudaFree(dyzscan));
	    CUDA_CHECK(cudaFree(yzhisto));
	}

	CUDA_CHECK(cudaMalloc(&dyzscan, sizeof(int) * yzncells));
	CUDA_CHECK(cudaMalloc(&yzhisto, sizeof(int) * yzncells));
	
	old_yzncells = yzncells;
    }

    size_t textureoffset = 0;
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texParticlesCLS, xyzuvw_copy, &texParticlesCLS.channelDesc, sizeof(float) * 6 * np));
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texScanYZ, dyzscan, &texScanYZ.channelDesc, sizeof(int) * ncells.y * ncells.z));
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texCountYZ, yzhisto, &texCountYZ.channelDesc, sizeof(int) * ncells.y * ncells.z));
    
    failuretest.reset(); 
    assert(failuretest.maxstripe != NULL);
    
    CUDA_CHECK(cudaMemcpyAsync(xyzuvw_copy, device_xyzuvw, sizeof(float) * 6 * np, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(yzhisto, 0, sizeof(int) * yzncells, stream));
    
    if (clists_perfmon)
	CUDA_CHECK(cudaEventRecord(evstart));

    {
	static const int ILP = 4;
	static const int SLOTS = 3;	
	static const int WARPS = 32;
	
	const int blocksize = 32 * WARPS;
	const int nblocks = (np + blocksize * ILP - 1)/ (blocksize * ILP);
	const int shmem_fp = sizeof(int) * ncells.y * ncells.z * SLOTS;
		
	*failuretest.maxstripe = 0;
	
	if (shmem_fp <= 32 * 1024)
	    yzhistogram<ILP, SLOTS, WARPS><<<nblocks, blocksize, sizeof(int) * ncells.y * ncells.z * SLOTS, stream>>>
		(np, 1 / rc, ncells, domainstart, yzcid,  loffsets, yzhisto, dyzscan, failuretest.dmaxstripe);
	else
	{
	    static const int SLOTS = 1;
	    
	    //printf("SHMEM: %.2f kB\n", (float)(sizeof(int) * ncells.y * ncells.z * SLOTS) / 1024.);
	    
	      yzhistogram<ILP, SLOTS, WARPS><<<nblocks, blocksize, sizeof(int) * ncells.y * ncells.z * SLOTS, stream>>>
		(np, 1 / rc, ncells, domainstart, yzcid,  loffsets, yzhisto, dyzscan, failuretest.dmaxstripe);
	}

	CUDA_CHECK(cudaPeekAtLastError());
    }

    CUDA_CHECK(cudaEventRecord(evacquire));
    
    {
	static const int ILP = 4;
	yzscatter<ILP><<<(np + 256 * ILP - 1) / (256 * ILP), 256, 0, stream>>>(loffsets, yzcid, np, outid);
    }
    
    CUDA_CHECK(cudaEventSynchronize(evacquire));

    {
	static const int YCPB = 2;

	xbufsize = *failuretest.maxstripe;

	int shmem_fp = sizeof(int) * (ncells.x  + 2 * xbufsize) * YCPB;

	if(shmem_fp < 48 * 1024)
	    xgather<YCPB><<< dim3(1, ncells.y / YCPB, ncells.z), dim3(32, YCPB), shmem_fp, stream>>>
		(outid, np, 1 / rc, ncells, domainstart, device_cellsstart, device_cellscount, device_xyzuvw, xbufsize,
		 host_order == NULL ? NULL : order);
	else
	{
	    static const int YCPB = 1;

	    shmem_fp = sizeof(int) * (ncells.x  + 2 * xbufsize) * YCPB;
	    
	    assert(shmem_fp < 48 * 1024);
	    
	    xgather<YCPB><<< dim3(1, ncells.y / YCPB, ncells.z), dim3(32, YCPB), shmem_fp, stream>>>
		(outid, np, 1 / rc, ncells, domainstart, device_cellsstart, device_cellscount, device_xyzuvw, xbufsize,
		 host_order == NULL ? NULL : order);
	}
    }
    
    if (clists_perfmon)
	CUDA_CHECK(cudaEventRecord(evscatter));
    
    if (!clists_robust)
    {
	failuretest.bufsize = xbufsize;
	CUDA_CHECK(cudaStreamAddCallback(stream, failuretest.callback_crash, &failuretest, 0));
    }
    else
    {
	CUDA_CHECK(cudaEventSynchronize(evacquire));

	if (*failuretest.maxstripe > xbufsize)
	{
	    //we should not be here anymore, after assignement at line 526
	    assert(false);
	    
	    CUDA_CHECK(cudaThreadSynchronize());
	    
	    printf("Ooops: maxstripe %d > bufsize %d.\nRecovering now...\n", *failuretest.maxstripe, xbufsize);
	    printf("density number223332 is %f\n", densitynumber);
	    const int xbufsize = *failuretest.maxstripe;
	    
	    xgather<1><<< dim3(1, ncells.y, ncells.z), dim3(32), sizeof(int) * (ncells.x  + 2 * xbufsize), stream>>>
		(outid, np, 1 / rc, ncells, domainstart, device_cellsstart, device_cellscount, device_xyzuvw, xbufsize,
		 host_order == NULL ? NULL : order);

	    cudaError_t status = cudaPeekAtLastError();

	    if (status != cudaSuccess)
	    {
		printf("Could not roll back. Aborting now.\n");
		abort();
	    }
	    else
		printf("Recovery succeeded.\n");
	}
    }

    if (clists_perfmon)
    {
	CUDA_CHECK(cudaEventRecord(evgather));
    
	CUDA_CHECK(cudaEventSynchronize(evgather));
   
	CUDA_CHECK(cudaPeekAtLastError());
	float tacquirems;
	CUDA_CHECK(cudaEventElapsedTime(&tacquirems, evstart, evacquire));
	float tscatterms;
	CUDA_CHECK(cudaEventElapsedTime(&tscatterms, evacquire, evscatter));
	float tgatherms;
	CUDA_CHECK(cudaEventElapsedTime(&tgatherms, evscatter, evgather));
	float ttotalms;
	CUDA_CHECK(cudaEventElapsedTime(&ttotalms, evstart, evgather));
    
	printf("acquiring time... %f ms\n", tacquirems);
	printf("scattering time... %f ms\n", tscatterms);
	printf("gathering time... %f ms\n", tgatherms);
	printf("total time ... %f ms\n", ttotalms);
	printf("one 2read-1write sweep should take about %.3f ms\n", 1e3 * np * 3 * 4/ (90.0 * 1024 * 1024 * 1024));
	printf("maxstripe was %d and bufsize is %d\n", *failuretest.maxstripe, xbufsize);
    }

    if (nonemptycells != NULL)
    {
	assert(nonemptycells->second != NULL);

	const int ntotcells = ncells.x * ncells.y * ncells.z;
	const int nonempties = copy_if(counting_iterator<int>(0), counting_iterator<int>(ntotcells), 
				       device_ptr<int>(device_cellscount), device_ptr<int>(nonemptycells->second), is_gzero())
	    - device_ptr<int>(nonemptycells->second);
	
	nonemptycells->first = nonempties;
    }

    if (host_order != NULL)
	CUDA_CHECK(cudaMemcpyAsync(host_order, order, sizeof(int) * np, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaUnbindTexture(texParticlesCLS));
    CUDA_CHECK(cudaUnbindTexture(texScanYZ));
    CUDA_CHECK(cudaUnbindTexture(texCountYZ));
}
