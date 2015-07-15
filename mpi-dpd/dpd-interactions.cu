/*
 *  dpd-interactions.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-04.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cassert>

#include <algorithm>

#include <cuda-dpd.h>

#include "dpd-interactions.h"

using namespace std;

ComputeInteractionsDPD::ComputeInteractionsDPD(MPI_Comm cartcomm): HaloExchanger(cartcomm, 0), local_trunk(0, 0, 0, 0)
{
    int myrank;
    MPI_CHECK(MPI_Comm_rank(cartcomm, &myrank));

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = (coords[c] + d[c] + dims[c]) % dims[c];

	int indx[3];
	for(int c = 0; c < 3; ++c)
	    indx[c] = min(coords[c], coordsneighbor[c]) * dims[c] + max(coords[c], coordsneighbor[c]);

	const int interrank_seed_base = indx[0] + dims[0] * dims[0] * (indx[1] + dims[1] * dims[1] * indx[2]);

	int interrank_seed_offset;

	{
	    const bool isplus =
		d[0] + d[1] + d[2] > 0 ||
		d[0] + d[1] + d[2] == 0 && (
		    d[0] > 0 || d[0] == 0 && (
			d[1] > 0 || d[1] == 0 && d[2] > 0
			)
		    );

	    const int mysign = 2 * isplus - 1;

	    int v[3] = { 1 + mysign * d[0], 1 + mysign * d[1], 1 + mysign *d[2] };

	    interrank_seed_offset = v[0] + 3 * (v[1] + 3 * v[2]);
	}

	const int interrank_seed = interrank_seed_base + interrank_seed_offset;

	interrank_trunks[i] = Logistic::KISS(390 + interrank_seed, interrank_seed  + 615, 12309, 23094);

	const int dstrank = dstranks[i];

	if (dstrank != myrank)
	    interrank_masks[i] = min(dstrank, myrank) == myrank;
	else
	{
	    int alter_ego = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	    interrank_masks[i] = min(i, alter_ego) == i;
	}
    }
}

void ComputeInteractionsDPD::local_interactions(const Particle * const p, const int n, Acceleration * const a,
						const int * const cellsstart, const int * const cellscount, cudaStream_t stream)
{
    NVTX_RANGE("DPD/local", NVTX_C5);

    if (n > 0)
	forces_dpd_cuda_nohost((float *)p, (float *)a, n,
			       cellsstart, cellscount,
			       1, XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN, aij, gammadpd,
			       sigma, 1. / sqrt(dt), local_trunk.get_float(), stream);
}

namespace BipsBatch
{
    __constant__ int start[27];

    struct BatchInfo
    {
	float * xdst, * xsrc, seed;
	int ndst, nsrc, mask, * cellstarts, * scattered_entries;
    };

    __constant__ BatchInfo batchinfos[26];

    __global__ void interactions_kernel(const float aij, const float gamma, const float sigmaf,	const int ndstall, const float * adst)
    {
	assert(ndstall <= gridDim.x * blockDim.x);
	assert(blockDim.x % warpSize == 0);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= start[26])
	    return;

	const int key9 = 9 * ((gid >= start[9]) + (gid >= start[18]));
	const int key3 = 3 * ((gid >= start[key9 + 3]) + (gid >= start[key9 + 6]));
	const int key1 = (gid >= start[key9 + key3 + 1]) + (gid >= start[key9 + key3 + 2]);

	const int code =  key9 + key3 + key1;
	const int pid = gid - start[entry];
	assert(code < 26);
	
	const BatchInfo info = batchinfos[entry];
	assert( pid < info.nd );

	const float xp = info.xdst[0 + pid * 6];
	const float yp = info.xdst[1 + pid * 6];
	const float zp = info.xdst[2 + pid * 6];
	const float up = info.xdst[3 + pid * 6];
	const float vp = info.xdst[4 + pid * 6];
	const float wp = info.xdst[5 + pid * 6];

	const int xcid = xp + XSIZE_SUBDOMAIN / 2;
	const int ycid = yp + YSIZE_SUBDOMAIN / 2;
	const int zcid = zp + ZSIZE_SUBDOMAIN / 2;

	const int m0 = 0 == (code + 2) % 3 - 1;
	const int m1 = 0 == (code / 3 + 2) % 3 - 1;
	const int m2 = 0 == (code / 9 + 2) % 3 - 1;

	float xforce = 0, yforce = 0, zforce = 0;

	for(int dz = -1; dz < 2; ++dz)
	    for(int dy = -1; dy < 2; ++dy)
		for(int dx = -1; dx < 2; ++dx)
		{
		    const int cid = ...;
		    const int cidnext = ...;

		    const int start = _ACCESS(cellstarts + cid);
		    const int stop = _ACCESS(cellstarts + cidnext);

		    for(int spid = start; spid < stop; ++spid)
		    {
			const float xq = _ACCESS(xsrc + 0 + spid * 6);
			const float yq = _ACCESS(xsrc + 1 + spid * 6);
			const float zq = _ACCESS(xsrc + 2 + spid * 6);
			const float uq = _ACCESS(xsrc + 3 + spid * 6);
			const float vq = _ACCESS(xsrc + 4 + spid * 6);
			const float wq = _ACCESS(xsrc + 5 + spid * 6);

			const float _xr = xp - xq;
			const float _yr = yp - yq;
			const float _zr = zp - zq;

			const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

			if (rij2 < 1)
			{
			    const float invrij = rsqrtf(rij2);

			    const float rij = rij2 * invrij;
			    const float argwr = 1.f - rij;
			    const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

			    const float xr = _xr * invrij;
			    const float yr = _yr * invrij;
			    const float zr = _zr * invrij;

			    const float rdotv =
				xr * (up - uq) +
				yr * (vp - vq) +
				zr * (wp - wq);

			    const int spid = s + l;
			    const int dpid = pid;

			    const int arg1 = mask * dpid + (1 - mask) * spid;
			    const int arg2 = mask * spid + (1 - mask) * dpid;
			    const float myrandnr = Logistic::mean0var1(seed, arg1, arg2);

			    const float strength = aij * argwr + (- gamma * wr * rdotv + sigmaf * myrandnr) * wr;

			    xforce += strength * xr;
			    yforce += strength * yr;
			    zforce += strength * zr;
			}
		    }
		}
	
	const int dstbase = 3 * info.scattered_indices[pid];

	atomicAdd(dst + dstbase + 0, xforce);
	atomicAdd(dst + dstbase + 1, yforce);
	atomicAdd(dst + dstbase + 2, zforce);
    }

    bool firstcall = true;

    void interactions(const float aij, const float gamma, const float sigma, const float invsqrtdt,
		      const BatchInfo infos[20], cudaStream_t stream, cudaEvent_t event, float * const acc)
    {
	if (firstcall)
	{
	    CUDA_CHECK(cudaFuncSetCacheConfig(interactions_kernel, cudaFuncCachePreferL1));
	    firstcall = false;
	}

	CUDA_CHECK(cudaMemcpyToSymbolAsync(batchinfos, infos, sizeof(BatchInfo) * 26, 0, cudaMemcpyHostToDevice, stream));

	int hstart[27];

	hstart[0] = 0;
	for(int i = 0; i < 26; ++i)
	    hstart[i + 1] = info[i].nd + hstart[i];

	CUDA_CHECK(cudaMemcpyToSymbolAsync(start, hstart, sizeof(hstart), 0, cudaMemcpyHostToDevice, stream));

	const int nhaloparticles = start_padded[26];

	CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));    

	interaction_kernel<<< (nhaloparticles + 127) / 128, 128, 0, stream>>>(aij, gamma, sigma * invsqrtdt, nthreads, acc);

	CUDA_CHECK(cudaPeekAtLastError());
    }
}

void ComputeInteractionsDPD::remote_interactions(const Particle * const p, const int n, Acceleration * const a, cudaStream_t stream)
{
    NVTX_RANGE("DPD/remote", NVTX_C3);

    CUDA_CHECK(cudaPeekAtLastError());    
	
    BipsBatch::BatchInfo infos[26];
    
    for(int i = 0; i < 26; ++i)
    {
	BipsBatch::BatchInfo entry = {
	    (float *)sendhalos[i].dbuf.data, (float *)recvhalos[i].dbuf.data, interrank_trunks[i].get_float(), 
	    sendhalos[i].dbuf.size, recvhalos[i].dbuf.size, interrank_masks[i], 
	    recvhalos[i].dcellstarts.data, sendhalos[i].scattered_entries.data  
	};
	
	infos[i] = entry;
    }
    
    assert(ctr == 20);

    BipsBatch::interactions(aij, gammadpd, sigma, 1. / sqrt(dt), infos, stream, evshiftrecvp, a);

    CUDA_CHECK(cudaPeekAtLastError());
}
