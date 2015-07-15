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

ComputeInteractionsDPD::ComputeInteractionsDPD(MPI_Comm cartcomm):
HaloExchanger(cartcomm, 0), local_trunk(0, 0, 0, 0)
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

    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaEventCreate(evremoteint + i, cudaEventDisableTiming));

    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaStreamCreate(streams + i));

    for(int i = 0, ctr = 1; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	const bool isface = abs(d[0]) + abs(d[1]) + abs(d[2]) == 1;

	code2stream[i] = i % 7;

	if (isface)
	{
	    code2stream[i] = ctr;
	    ctr++;
	}
    }
}

namespace LocalDPD
{
    __global__ void merge(const float * const src, float * const dst, const int n)
    {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;

	if (gid < n)
	    dst[gid] += src[gid];
    }
}

void ComputeInteractionsDPD::local_interactions(const Particle * const p, const int n, Acceleration * const a,
						const int * const cellsstart, const int * const cellscount, cudaStream_t stream)
{
    NVTX_RANGE("DPD/local", NVTX_C5);

    if (n > 0)
    {
	acc_local.resize(n);

	forces_dpd_cuda_nohost((float *)p, (float *)acc_local.data, n,
			       cellsstart, cellscount,
			       1, XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN, aij, gammadpd,
			       sigma, 1. / sqrt(dt), local_trunk.get_float(), stream);

	LocalDPD::merge<<< (n * 3 + 127) / 128, 128, 0, stream >>>((float *)acc_local.data, (float *)a, 3 * n);
    }
}

namespace RemoteDPD
{
    int npackedparticles;

    __constant__ int packstarts[27];
    __constant__ int * scattered_indices[26];
    __constant__ Acceleration * remote_accelerations[26];

    __global__ void merge_all(Acceleration * const alocal, const int nlocal, const int nremote)
    {
	assert(blockDim.x * gridDim.x >= nremote);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= packstarts[26])
	    return;

	const int key9 = 9 * ((gid >= packstarts[9]) + (gid >= packstarts[18]));
	const int key3 = 3 * ((gid >= packstarts[key9 + 3]) + (gid >= packstarts[key9 + 6]));
	const int key1 = (gid >= packstarts[key9 + key3 + 1]) + (gid >= packstarts[key9 + key3 + 2]);
	const int idpack = key9 + key3 + key1;

	assert(idpack >= 0 && idpack < 26);
	assert(gid >= packstarts[idpack] && gid < packstarts[idpack + 1]);

	const int offset = gid - packstarts[idpack];

	int pid = scattered_indices[idpack][offset];

	if (!(pid >= 0 && pid < nlocal))
	    cuda_printf("oooooops pid is %d whereas nlocal is %d\n", pid, nlocal);
	assert(pid >= 0 && pid < nlocal);

	Acceleration a = remote_accelerations[idpack][offset];

	for(int c = 0; c < 3; ++c)
	    assert(!isnan(a.a[c]));

	for(int c = 0; c < 3; ++c)
	    atomicAdd(& alocal[pid].a[c], a.a[c]);
    }

    __global__ void merge_accelerations(const Acceleration * const aremote, const int nremote,
					Acceleration * const alocal, const int nlocal,
					const Particle * premote, const Particle * plocal,
					const int * const scattered_entries, int rank)
    {
	assert(blockDim.x * gridDim.x >= nremote);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= nremote)
	    return;

	int pid = scattered_entries[ gid ];
	assert(pid >= 0 && pid < nlocal);

	Acceleration a = aremote[gid];

#ifndef NDEBUG
	Particle p1 = plocal[pid];
	Particle p2 = premote[gid];

	for(int c = 0; c < 3; ++c)
	{
	    assert(p1.x[c] == p2.x[c]);
	    assert(p1.x[c] == p2.x[c]);
	}

	for(int c = 0; c < 3; ++c)
	{
	    if (isnan(a.a[c]))
		printf("rank %d) oouch gid %d %f out of %d remote entries going to pid %d of %d particles\n",
		       rank, gid, a.a[c], nremote, pid, nlocal);

	    assert(!isnan(a.a[c]));
	}
#endif
	for(int c = 0; c < 3; ++c)
	{
	    atomicAdd(& alocal[pid].a[c], a.a[c]);

	    assert(!isnan(a.a[c]));
	}
    }
}

namespace BipsBatch
{
    __constant__ int start[21];

    struct BatchInfo
    {
	float * xdst, * adst, * xsrc, seed;
	int ndst, nsrc, mask; //, cellsdirection, * cellstarts;
    };

    __constant__ BatchInfo batchinfos[20];

    __global__ void interactions_kernel(const float aij, const float gamma, const float sigmaf,	const int ndstall)
    {
	assert(ndstall <= gridDim.x * blockDim.x);
	assert(blockDim.x % warpSize == 0);

	const int tid = threadIdx.x & 31;
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= ndstall)
	    return;

	//find my halo bag
	const int key4 = 4 * ((gid >= start[4]) + (gid >= start[8]) + (gid >= start[12]) + (gid >= start[16]));
	const int key2 = key4 + 2 * (gid >= start[key4 + 2]);
	const int key = key2 + (gid >= start[key2 + 1]);
	assert(key4 + 4 < 21 && key2 + 2 < 21 && key + 1 < 21);

	const int entry = key;
	assert(entry < 20);

	const int pid = gid - start[entry];
	assert(tid == pid % warpSize);

	const BatchInfo info = batchinfos[entry];

	const int nd = info.ndst;
	const int ns = info.nsrc;
	const int mask = info.mask;
	const float seed = info.seed;
	const float * const xdst = info.xdst;
	const float * const xsrc = info.xsrc;
	float * const adst = info.adst;

	const bool valid = pid < nd;

	float xp, yp, zp, up, vp, wp;

	if (valid)
	{
	    xp = xdst[0 + pid * 6];
	    yp = xdst[1 + pid * 6];
	    zp = xdst[2 + pid * 6];
	    up = xdst[3 + pid * 6];
	    vp = xdst[4 + pid * 6];
	    wp = xdst[5 + pid * 6];
	}

	float xforce = 0, yforce = 0, zforce = 0;

	for(int s = 0; s < ns; s += warpSize)
	{
	    float my_xq, my_yq, my_zq, my_uq, my_vq, my_wq;

	    const int batchsize = min(warpSize, ns - s);

	    if (tid < batchsize)
	    {
		my_xq = __ldg(xsrc + 0 + (tid + s) * 6);
		my_yq = __ldg(xsrc + 1 + (tid + s) * 6);
		my_zq = __ldg(xsrc + 2 + (tid + s) * 6);
		my_uq = __ldg(xsrc + 3 + (tid + s) * 6);
		my_vq = __ldg(xsrc + 4 + (tid + s) * 6);
		my_wq = __ldg(xsrc + 5 + (tid + s) * 6);
	    }


#pragma unroll 8
	    for(int l = 0; l < batchsize; ++l)
	    {
		const float xq = __shfl(my_xq, l);
		const float yq = __shfl(my_yq, l);
		const float zq = __shfl(my_zq, l);
		const float uq = __shfl(my_uq, l);
		const float vq = __shfl(my_vq, l);
		const float wq = __shfl(my_wq, l);

		{
		    const float _xr = xp - xq;
		    const float _yr = yp - yq;
		    const float _zr = zp - zq;

		    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

		    const float invrij = rsqrtf(rij2);

		    const float rij = rij2 * invrij;
		    const float argwr = max((float)0, 1 - rij);
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

		    //if (valid && spid < np_src)
		    {
			xforce += strength * xr;
			yforce += strength * yr;
			zforce += strength * zr;
		    }
		}
	    }
	}

	if (valid)
	{
	    assert(!isnan(xforce));
	    assert(!isnan(yforce));
	    assert(!isnan(zforce));

	    adst[0 + 3 * pid] = xforce;
	    adst[1 + 3 * pid] = yforce;
	    adst[2 + 3 * pid] = zforce;
	}
    }

    bool firstcall = true;

    void interactions(const float aij, const float gamma, const float sigma, const float invsqrtdt,
		      const BatchInfo infos[20], cudaStream_t stream)
    {
	if (firstcall)
	{
	    CUDA_CHECK(cudaFuncSetCacheConfig(interactions_kernel, cudaFuncCachePreferL1));
	    firstcall = false;
	}

	CUDA_CHECK(cudaMemcpyToSymbolAsync(batchinfos, infos, sizeof(BatchInfo) * 20, 0, cudaMemcpyHostToDevice, stream));

	int count_padded[20];
	for(int i = 0; i < 20; ++i)
	    count_padded[i] = 32 * ((infos[i].ndst + 31) / 32);

	int start_padded[21];
	for(int i = 0, s = 0; i < 21; ++i)
	    s = count_padded[i] + (start_padded[i] = s);

	CUDA_CHECK(cudaMemcpyToSymbolAsync(start, start_padded, sizeof(start_padded), 0, cudaMemcpyHostToDevice, stream));

	const int nthreads = start_padded[20];
	assert(nthreads % 32 == 0);

	interactions_kernel<<<(nthreads + 31) / 32, 32, 0, stream>>>(aij, gamma, sigma * invsqrtdt, nthreads);
	//printf("spawiniing %d cuda blocks\n", (nthreads + 31) / 32);

	CUDA_CHECK(cudaPeekAtLastError());
    }
}

void ComputeInteractionsDPD::remote_interactions(const Particle * const p, const int n, Acceleration * const a, cudaStream_t stream)
{
    CUDA_CHECK(cudaPeekAtLastError());

    {
	NVTX_RANGE("DPD/remote", NVTX_C3);

	for(int i = 0; i < 7; ++i)
	    CUDA_CHECK(cudaStreamWaitEvent(streams[i], evshiftrecvp, 0));

	static const bool batchdbips = true;

	for(int pass = 0; pass < 2; ++pass)
	{
	    const bool face_pass = pass == 0;

	    for(int i = 0; i < 26; ++i)
	    {
		int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

		const bool isface = abs(d[0]) + abs(d[1]) + abs(d[2]) == 1;

		if (isface != face_pass)
		    continue;

		const float interrank_seed = interrank_trunks[i].get_float();

		const int nd = sendhalos[i].dbuf.size;
		const int ns = recvhalos[i].dbuf.size;

		acc_remote[i].resize(nd);

		if (nd == 0)
		    continue;

		cudaStream_t mystream = streams[code2stream[i]];

#ifndef NDEBUG
		//fill acc entries with nan
		CUDA_CHECK(cudaMemsetAsync(acc_remote[i].data, 0xff, sizeof(Acceleration) * acc_remote[i].size, mystream));
#endif
		if (ns == 0)
		{
		    CUDA_CHECK(cudaMemsetAsync((float *)acc_remote[i].data, 0, nd * sizeof(Acceleration), mystream));
		    continue;
		}

		if (isface)//sendhalos[i].dcellstarts.size * recvhalos[i].dcellstarts.size > 1 && nd * ns > 10 * 10)
		{

		    texDC[i].acquire(sendhalos[i].dcellstarts.data, sendhalos[i].dcellstarts.capacity);
		    texSC[i].acquire(recvhalos[i].dcellstarts.data, recvhalos[i].dcellstarts.capacity);
		    texSP[i].acquire((float2*)recvhalos[i].dbuf.data, recvhalos[i].dbuf.capacity * 3);

		    forces_dpd_cuda_bipartite_nohost(mystream, (float2 *)sendhalos[i].dbuf.data, nd, texDC[i].texObj, texSC[i].texObj, texSP[i].texObj,
						     ns, halosize[i], aij, gammadpd, sigma / sqrt(dt), interrank_seed, interrank_masks[i],
						     (float *)acc_remote[i].data);
		}
		else
		    if (!batchdbips)
			directforces_dpd_cuda_bipartite_nohost(
			    (float *)sendhalos[i].dbuf.data, (float *)acc_remote[i].data, nd,
			    (float *)recvhalos[i].dbuf.data, ns,
			    aij, gammadpd, sigma, 1. / sqrt(dt), interrank_seed, interrank_masks[i], mystream);

	    }
	}

	if (batchdbips)
	{
	    BipsBatch::BatchInfo infos[20];

	    int ctr = 0;

	    for(int i = 0; i < 26; ++i)
	    {
		int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

		const bool isface = abs(d[0]) + abs(d[1]) + abs(d[2]) == 1;

		if (isface)
		    continue;

		const bool isedge = abs(d[0]) + abs(d[1]) + abs(d[2]) == 1;
		const bool edgedir = 0 + (abs(d[1]) == 0) * 1 + (abs(d[2]) == 0) * 2;
		
		BipsBatch::BatchInfo entry = {
		    (float *)sendhalos[i].dbuf.data, (float *)acc_remote[i].data, (float *)recvhalos[i].dbuf.data,
		    interrank_trunks[i].get_float(), sendhalos[i].dbuf.size, recvhalos[i].dbuf.size, interrank_masks[i] //,
		    //isedge ? edgedir : -1, recvhalos[i].dcellstarts.data  
		};

		infos[ ctr++ ] = entry;
	    }

	    assert(ctr == 20);

	    BipsBatch::interactions(aij, gammadpd, sigma, 1. / sqrt(dt), infos, streams[0]);
	}

	for(int i = 0; i < 7; ++i)
	    CUDA_CHECK(cudaEventRecord(evremoteint[i], streams[i]));

	CUDA_CHECK(cudaPeekAtLastError());
    }

    {
	NVTX_RANGE("DPD/merge", NVTX_C6);

	{
	    static int packstarts[27];

	    packstarts[0] = 0;
	    for(int i = 0, s = 0; i < 26; ++i)
		packstarts[i + 1] =  (s += acc_remote[i].size * (sendhalos[i].expected > 0));

	    RemoteDPD::npackedparticles = packstarts[26];

	    if (!is_mps_enabled)
		CUDA_CHECK(cudaMemcpyToSymbolAsync(RemoteDPD::packstarts, packstarts,
						   sizeof(packstarts), 0, cudaMemcpyHostToDevice, stream));
	    else
		CUDA_CHECK(cudaMemcpyToSymbol(RemoteDPD::packstarts, packstarts,
					      sizeof(packstarts), 0, cudaMemcpyHostToDevice));
	}

	{
	    static int * scattered_indices[26];
	    for(int i = 0; i < 26; ++i)
		scattered_indices[i] = sendhalos[i].scattered_entries.data;

	    if (!is_mps_enabled)
		CUDA_CHECK(cudaMemcpyToSymbolAsync(RemoteDPD::scattered_indices, scattered_indices,
						   sizeof(scattered_indices), 0, cudaMemcpyHostToDevice, stream));
	    else
		CUDA_CHECK(cudaMemcpyToSymbol(RemoteDPD::scattered_indices, scattered_indices,
					      sizeof(scattered_indices), 0, cudaMemcpyHostToDevice));
	}

	{
	    static Acceleration * remote_accelerations[26];

	    for(int i = 0; i < 26; ++i)
		remote_accelerations[i] = acc_remote[i].data;

	    if (!is_mps_enabled)
		CUDA_CHECK(cudaMemcpyToSymbolAsync(RemoteDPD::remote_accelerations, remote_accelerations,
						   sizeof(remote_accelerations), 0, cudaMemcpyHostToDevice, stream));
	    else
		CUDA_CHECK(cudaMemcpyToSymbol(RemoteDPD::remote_accelerations, remote_accelerations,
					      sizeof(remote_accelerations), 0, cudaMemcpyHostToDevice));
	}

	for(int i = 0; i < 7; ++i)
	    CUDA_CHECK(cudaStreamWaitEvent(stream, evremoteint[i], 0));

#if 1
	RemoteDPD::merge_all<<< (RemoteDPD::npackedparticles + 127) / 128, 128, 0, stream >>>(a, n, RemoteDPD::npackedparticles);
#else
	for(int i = 0; i < 26; ++i)
	{
	    const int nd = acc_remote[i].size;

	    if (nd > 0)
		RemoteDPD::merge_accelerations<<<(nd + 127) / 128, 128, 0, streams[code2stream[i]]>>>
		    (acc_remote[i].data, nd, a, n, sendhalos[i].dbuf.data, p, sendhalos[i].scattered_entries.data, myrank);
	}
#endif
	CUDA_CHECK(cudaPeekAtLastError());
    }
}

ComputeInteractionsDPD::~ComputeInteractionsDPD()
{
    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaStreamDestroy(streams[i]));

    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaEventDestroy(evremoteint[i]));
}
