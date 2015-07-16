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

    __global__ void /*__launch_bounds__(128, 12)*/
    interaction_kernel(const float aij, const float gamma, const float sigmaf,
		       const int ndstall, float * const adst, const int sizeadst)
    {
#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined!
#define _ACCESS(x) (*(x))
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

	assert(ndstall <= gridDim.x * blockDim.x);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= start[26])
	    return;

	int code, dpid;

	{
	    const int key9 = 9 * ((gid >= start[9]) + (gid >= start[18]));
	    const int key3 = 3 * ((gid >= start[key9 + 3]) + (gid >= start[key9 + 6]));
	    const int key1 = (gid >= start[key9 + key3 + 1]) + (gid >= start[key9 + key3 + 2]);

	    code =  key9 + key3 + key1;
	    dpid = gid - start[code];
	    assert(code < 26);
	}

	const BatchInfo info = batchinfos[code];

	if (dpid >= info.ndst)
	    return;

	const float xp = info.xdst[0 + dpid * 6];
	const float yp = info.xdst[1 + dpid * 6];
	const float zp = info.xdst[2 + dpid * 6];
	const float up = info.xdst[3 + dpid * 6];
	const float vp = info.xdst[4 + dpid * 6];
	const float wp = info.xdst[5 + dpid * 6];

	const float * const xsrc = info.xsrc;
	const int mask = info.mask;
	const float seed = info.seed;

	const int dstbase = 3 * info.scattered_entries[dpid];
	assert(dstbase < sizeadst * 3);

	int xcells, ycells, basecid, xstencilsize, ystencilsize, stencilsize;

	{
	    const int m0 = 0 == (code + 2) % 3 - 1;
	    const int m1 = 0 == (code / 3 + 2) % 3 - 1;
	    const int m2 = 0 == (code / 9 + 2) % 3 - 1;

	    xcells = 1 + m0 * (XSIZE_SUBDOMAIN - 1);
	    ycells = 1 + m1 * (YSIZE_SUBDOMAIN - 1);
	    const int zcells = 1 + m2 * (ZSIZE_SUBDOMAIN - 1);

	    const int xcid = (int)(xp + XSIZE_SUBDOMAIN / 2);
	    const int ycid = (int)(yp + YSIZE_SUBDOMAIN / 2);
	    const int zcid = (int)(zp + ZSIZE_SUBDOMAIN / 2);
	    assert(xcid >= 0 && ycid >= 0 && zcid >= 0);

	    const int xbasecid = m0 * max(0, xcid - 1);
	    const int ybasecid = m1 * max(0, ycid - 1);
	    const int zbasecid = m2 * max(0, zcid - 1);
	    basecid = xbasecid + xcells * (ybasecid + ycells * zbasecid);

	    xstencilsize = 1 + m0 * (min(xcells, xcid + 2) -1 - xbasecid);
	    ystencilsize = 1 + m1 * (min(ycells, ycid + 2) -1 - ybasecid);
	    const int zstencilsize = 1 + m2 * (min(zcells, zcid + 2) -1 - zbasecid);

	    stencilsize = xstencilsize * ystencilsize * zstencilsize;
	    assert(stencilsize > 0);
	}

	float xforce = 0, yforce = 0, zforce = 0;

	int itstencil = -1, countp = 0, spid;

	asm volatile ("//start loop");

	do
	{
	    while (countp == 0)
	    {
		++itstencil;

		if (itstencil >= stencilsize)
		    goto endloop;

		const int tmp = itstencil / xstencilsize;

		const int currcid = basecid + (itstencil % xstencilsize) +
		    xcells * ((tmp % ystencilsize) + ycells * (tmp / ystencilsize));

		spid = _ACCESS(info.cellstarts + currcid);
		assert(spid >= 0);

		countp = _ACCESS(info.cellstarts + currcid + 1) - spid;
		assert(countp >= 0);
	    }

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

		const int arg1 = mask * dpid + (1 - mask) * spid;
		const int arg2 = mask * spid + (1 - mask) * dpid;
		const float myrandnr = Logistic::mean0var1(seed, arg1, arg2);

		const float strength = aij * argwr + (- gamma * wr * rdotv + sigmaf * myrandnr) * wr;

		xforce += strength * xr;
		yforce += strength * yr;
		zforce += strength * zr;

/*		atomicAdd(adst + dstbase + 0, strength * xr);
		atomicAdd(adst + dstbase + 1, strength * yr);
		atomicAdd(adst + dstbase + 2, strength * zr);
*/	    }

	    ++spid;
	    --countp;
	}
	while(true);

    endloop:

	atomicAdd(adst + dstbase + 0, xforce);
	atomicAdd(adst + dstbase + 1, yforce);
	atomicAdd(adst + dstbase + 2, zforce);

#undef _ACCESS
    }

    bool firstcall = true;

    void interactions(const float aij, const float gamma, const float sigma, const float invsqrtdt,
		      const BatchInfo infos[20], cudaStream_t stream, cudaEvent_t event, float * const acc, const int n)
    {
	if (firstcall)
	{
	    CUDA_CHECK(cudaFuncSetCacheConfig(interaction_kernel, cudaFuncCachePreferL1));
	    firstcall = false;
	}

	CUDA_CHECK(cudaMemcpyToSymbolAsync(batchinfos, infos, sizeof(BatchInfo) * 26, 0, cudaMemcpyHostToDevice, stream));

	int hstart_padded[27];

	hstart_padded[0] = 0;
	for(int i = 0; i < 26; ++i)
	    hstart_padded[i + 1] = hstart_padded[i] + 32 * ((infos[i].ndst + 31)/ 32) ;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(start, hstart_padded, sizeof(hstart_padded), 0, cudaMemcpyHostToDevice, stream));

	const int nthreads = hstart_padded[26];

	CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));

	interaction_kernel<<< (nthreads + 127) / 128, 128, 0, stream>>>(aij, gamma, sigma * invsqrtdt, nthreads, acc, n);
//printf("launching %d blocks\n", nthreads / 128);
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

    BipsBatch::interactions(aij, gammadpd, sigma, 1. / sqrt(dt), infos, stream, evshiftrecvp, (float *)a, n);

    CUDA_CHECK(cudaPeekAtLastError());
}
