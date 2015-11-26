/*
 *  redistribute-particles.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-02-09.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cassert>
#include <vector>
#include <algorithm>

#include "common-kernels.h"
#include "scan.h"
#include "redistribute-particles.h"

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

using namespace std;

namespace RedistributeParticlesKernels
{
    __constant__ RedistributeParticles::PackBuffer pack_buffers[27];

    __constant__ RedistributeParticles::UnpackBuffer unpack_buffers[27];

    __device__ int pack_count[27], pack_start_padded[28];

    __constant__ int unpack_start[28], unpack_start_padded[28];

    __device__ bool failed;

    int ntexparticles = 0;
    float2 * texparticledata;
    texture<float, cudaTextureType1D> texAllParticles;
    texture<float2, cudaTextureType1D> texAllParticlesFloat2;

#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined! assuming 350
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

    __global__ void setup()
    {
	if (threadIdx.x == 0)
	    failed = false;

	if (threadIdx.x < 27)
	    pack_count[threadIdx.x] = 0;
    }

    __global__ void scatter_halo_indices_pack(const int np)
    {
	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid < np)
	{
	    float xp[3];
	    for(int c = 0; c < 3; ++c)
		xp[c] = tex1Dfetch(texAllParticles, 6 * pid + c);

	    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (xp[c] >= -L[c]/2) + (xp[c] >= L[c]/2)) % 3;

	    const int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
	    assert(code >= 0 && code < 27);

	    if (code > 0)
	    {
		const int entry = atomicAdd(pack_count + code, 1);

		if (entry < pack_buffers[code].capacity)
		    pack_buffers[code].scattered_indices[entry] = pid;
	    }
	}
    }

    __global__ void tiny_scan(const int nparticles, const int bulkcapacity, int * const packsizes, bool * const failureflag)
    {
	assert(blockDim.x > 27 && gridDim.x == 1);

	const int tid = threadIdx.x;

	int myval = 0, mycount = 0;

	if (tid < 27)
	{
	    myval = mycount = pack_count[threadIdx.x];
	    if (tid > 0)
		packsizes[tid] = mycount;

	    if (mycount > pack_buffers[tid].capacity)
	    {
		failed = true;
		*failureflag = true;
	    }
	}

	//myval = 32 * ((myval + 31) / 32);

	for(int L = 1; L < 32; L <<= 1)
	    myval += (tid >= L) * __shfl_up(myval, L) ;

	if (tid < 28)
	    pack_start_padded[tid] = myval - mycount;

	if (tid == 26)
	{
	    pack_start_padded[tid + 1] = myval;

	    const int nbulk = nparticles - myval;
	    packsizes[0] = nbulk;

	    if (nbulk > bulkcapacity)
	    {
		failed = true;
		*failureflag = true;
	    }
	}
    }

#ifndef NDEBUG
    __global__ void check_scan()
    {
	assert(blockDim.x == 1 && gridDim.x == 1);

	for(int i = 1; i < 28; ++i)
	    assert(pack_start_padded[i - 1] <= pack_start_padded[i]);
    }
#endif

    __global__ void pack(const int nparticles, const int nfloat2s)
    {

	assert(blockDim.x * gridDim.x >= nfloat2s);

	if (failed)
	    return;

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	const int slot = gid / 3;

	const int tid = threadIdx.x;

	__shared__ int start[28]; //, count[27];

	if (tid < 28)
	    start[tid] = pack_start_padded[tid];

	//if (tid < 27)
	//   count[tid] = pack_count[tid];

	__syncthreads();

	const int key9 = 9 * (slot >= start[9]) + 9 * (slot >= start[18]);
	const int key3 = 3 * (slot >= start[key9 + 3]) + 3 * (slot >= start[key9 + 6]);
	const int key1 = (slot >= start[key9 + key3 + 1]) + (slot >= start[key9 + key3 + 2]);

	const int idpack = key9 + key3 + key1;

	if (slot >= start[27])
	    return;

	const int offset = slot - start[idpack];

	//if (offset >= count[idpack])
	//    return;

	assert (offset >= 0 && offset < pack_buffers[idpack].capacity);

	const int pid = _ACCESS(pack_buffers[idpack].scattered_indices + offset);
	assert(pid < nparticles && pid >= 0);

	const int c = gid % 3;
	const int d = c + 3 * offset;
	assert (d < pack_buffers[idpack].capacity * 3);

	pack_buffers[idpack].buffer[d] = tex1Dfetch(texAllParticlesFloat2, c + 3 * pid);
    }

    __global__ void subindex_remote(const uint nparticles_padded,
				    const uint nparticles, int * const partials, float2 * const dstbuf, uchar4 * const subindices)
    {
	assert(blockDim.x * gridDim.x >= nparticles_padded && blockDim.x == 128);

	const uint warpid = threadIdx.x >> 5;

	const uint localbase = 32 * (warpid + 4 * blockIdx.x);

	if (localbase >= nparticles_padded)
	    return;

	const uint key9 = 9 * (localbase >= unpack_start_padded[9]) + 9 * (localbase >= unpack_start_padded[18]);
	const uint key3 = 3 * (localbase >= unpack_start_padded[key9 + 3]) + 3 * (localbase >= unpack_start_padded[key9 + 6]);
	const uint key1 = (localbase >= unpack_start_padded[key9 + key3 + 1]) + (localbase >= unpack_start_padded[key9 + key3 + 2]);
	const int code = key9 + key3 + key1;
	assert(code >= 1 && code < 28);
	assert(localbase >= unpack_start_padded[code] && localbase < unpack_start_padded[code + 1]);

	const int unpackbase = localbase - unpack_start_padded[code];
	assert (unpackbase >= 0);
	assert(unpackbase < unpack_buffers[code].capacity);

	const uint nunpack = min(32, unpack_start[code + 1] - unpack_start[code] - unpackbase);

	if (nunpack == 0)
	    return;

	float2 data0, data1, data2;

	read_AOS6f(unpack_buffers[code].buffer + 3 * unpackbase, nunpack, data0, data1, data2);

	const uint laneid = threadIdx.x & 0x1f;

	int xcid, ycid, zcid, subindex;

	if (laneid < nunpack)
	{
	    data0.x += XSIZE_SUBDOMAIN * ((code + 1) % 3 - 1);
	    data0.y += YSIZE_SUBDOMAIN * ((code / 3 + 1) % 3 - 1);
	    data1.x += ZSIZE_SUBDOMAIN * ((code / 9 + 1) % 3 - 1);

	    xcid = (int)floor((double)data0.x + XSIZE_SUBDOMAIN / 2);
	    ycid = (int)floor((double)data0.y + YSIZE_SUBDOMAIN / 2);
	    zcid = (int)floor((double)data1.x + ZSIZE_SUBDOMAIN / 2);

	    assert(xcid >= 0 && xcid < XSIZE_SUBDOMAIN &&
		   ycid >= 0 && ycid < YSIZE_SUBDOMAIN &&
		   zcid >= 0 && zcid < ZSIZE_SUBDOMAIN );

	    const int cid = xcid + XSIZE_SUBDOMAIN * (ycid + YSIZE_SUBDOMAIN * zcid);

	    subindex = atomicAdd(partials + cid, 1);

	    assert(subindex < 255);
	}

	const uint dstbase = unpack_start[code] + unpackbase;

	write_AOS6f(dstbuf + 3 * dstbase, nunpack, data0, data1, data2);

	if (laneid < nunpack)
	    subindices[dstbase + laneid] = make_uchar4(xcid, ycid, zcid, subindex);
    }

    __global__ void scatter_indices(const bool remote, const uchar4 * const subindices, const int nparticles,
				    const int * const starts, uint * const scattered_indices, const int nscattered)
    {
	assert(blockDim.x * gridDim.x >= nparticles);

	uint pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= nparticles)
	    return;

	const uchar4 entry = subindices[pid];

	const int subindex = entry.w;

	if (subindex != 255)
	{
	    const int cid = entry.x + XSIZE_SUBDOMAIN * (entry.y + YSIZE_SUBDOMAIN * entry.z);
	    const int base = _ACCESS(starts + cid);

	    pid |= remote << 31;

	    assert(base + subindex < nscattered);

	    //if (pid == 0)
	    //	printf("pid %d: base: %d subindex %d cid %d\n", pid, base, subindex, cid);

	    scattered_indices[base + subindex] = pid;
	}
    }

    __forceinline__ __device__ void xchg_aos2f(const int srclane0, const int srclane1, const int start, float& s0, float& s1)
    {
	const float t0 = __shfl(s0, srclane0);
	const float t1 = __shfl(s1, srclane1);

	s0 = start == 0 ? t0 : t1;
	s1 = start == 0 ? t1 : t0;

	s1 = __shfl_xor(s1, 1);
    }

    __forceinline__ __device__ void xchg_aos4f(const int srclane0, const int srclane1, const int start, float3& s0, float3& s1)
    {
	xchg_aos2f(srclane0, srclane1, start, s0.x, s1.x);
	xchg_aos2f(srclane0, srclane1, start, s0.y, s1.y);
	xchg_aos2f(srclane0, srclane1, start, s0.z, s1.z);
    }

    __global__ void gather_particles(const uint * const scattered_indices,
				     const float2 * const remoteparticles, const int nremoteparticles,
				     const int noldparticles,
				     const int nparticles,
				     float2 * const dstbuf,
				     float4 * const xyzouvwo,
				     ushort4 * const xyzo_half)
    {
	assert(blockDim.x == 128);

	const int warpid = threadIdx.x >> 5;
	const int tid = threadIdx.x & 0x1f;

	const int base = 32 * (warpid + 4 * blockIdx.x);
	const int pid = base + tid;

	const bool valid = (pid < nparticles);

	uint spid;

	if (valid)
	    spid = scattered_indices[pid];

	float2 data0, data1, data2;

	if (valid)
	{
	    const bool remote = (spid >> 31) & 1;

	    spid &= ~(1 << 31);

	    if (remote)
	    {
		assert(spid < nremoteparticles);
		data0 = _ACCESS(remoteparticles + 0 + 3 * spid);
		data1 = _ACCESS(remoteparticles + 1 + 3 * spid);
		data2 = _ACCESS(remoteparticles + 2 + 3 * spid);
	    }
	    else
	    {
		if (spid >= noldparticles)
		    cuda_printf("ooops pid %d spid %d noldp%d\n", pid, spid, noldparticles);

		assert(spid < noldparticles);
		data0 = tex1Dfetch(texAllParticlesFloat2, 0 + 3 * spid);
		data1 = tex1Dfetch(texAllParticlesFloat2, 1 + 3 * spid);
		data2 = tex1Dfetch(texAllParticlesFloat2, 2 + 3 * spid);
	    }
	}

	const int nsrc = min(32, nparticles - base);


	{
	    //if (tid < nsrc) {xyzouvwo[2 * (base + tid) + 0] = make_float4(data0.x, data0.y, data1.x, 0);
	    //	xyzouvwo[2 * (base + tid) + 1] = make_float4(data1.y, data2.x, data2.y, 0);}


	    const int srclane0 = (32 * ((tid) & 0x1) + tid) >> 1;
	    const int srclane1 = (32 * ((tid + 1) & 0x1) + tid) >> 1;
	    const int start = tid % 2;
	    const int destbase = 2 * base;

	    float3 s0 = make_float3(data0.x, data0.y, data1.x);
	    float3 s1 = make_float3(data1.y, data2.x, data2.y);

	    xchg_aos4f(srclane0, srclane1, start, s0, s1);

	    if (tid < 2 * nsrc)
		xyzouvwo[destbase + tid] = make_float4(s0.x, s0.y, s0.z, 0);

	    if (tid + 32 < 2 * nsrc)
		xyzouvwo[destbase + tid + 32] = make_float4(s1.x, s1.y, s1.z, 0);
	}

	if (tid < nsrc)
	{
	    xyzo_half[base + tid] = make_ushort4(
		__float2half_rn(data0.x),
		__float2half_rn(data0.y),
		__float2half_rn(data1.x), 0);
	}

	write_AOS6f(dstbuf + 3 * base, nsrc, data0, data1, data2);
    }

#ifndef NDEBUG
    __global__ void check(const int * const starts, const int * const counts, const Particle * const p, const int np)
    {
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN* ZSIZE_SUBDOMAIN)
	    return;

	const int count = counts[gid];
	const int start = starts[gid];


	const int xcid = gid % XSIZE_SUBDOMAIN;
	const int ycid = (gid / XSIZE_SUBDOMAIN) % YSIZE_SUBDOMAIN;
	const int zcid = gid / XSIZE_SUBDOMAIN / YSIZE_SUBDOMAIN ;

	const float xmin[3] = { xcid - XSIZE_SUBDOMAIN / 2,
				ycid - YSIZE_SUBDOMAIN / 2,
				zcid - ZSIZE_SUBDOMAIN / 2 };

	for(int i = 0; i < count; ++i)
	{
	    const int pid = start + i;

	    assert(pid < np && pid >= 0);

	    for(int c = 0; c < 3; ++c)
	    {
		assert(!isnan(p[pid].x[c]));

		if (!(p[pid].x[c] >= xmin[c] && p[pid].x[c] < xmin[c] + 1))
		{
		    printf("oooops pid %d c %d is %f of cell %d with count %d at entry %d not win [%f, %f[\n", pid, c, p[pid].x[c], gid, count, i,
			   xmin[c], xmin[c] + 1);
		}

		assert(p[pid].x[c] >= xmin[c] && p[pid].x[c] < xmin[c] + 1);
	    }
	}
    }
#endif

#undef _ACCESS
}

RedistributeParticles::RedistributeParticles(MPI_Comm _cartcomm):
failure(1), packsizes(27), nactiveneighbors(26), firstcall(true),
compressed_cellcounts(XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN),
subindices(1.5 * numberdensity * XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN),
subindices_remote(1.5 * numberdensity * (XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN -
					 (XSIZE_SUBDOMAIN - 2) * (YSIZE_SUBDOMAIN - 2) * (ZSIZE_SUBDOMAIN - 2)))
{
    safety_factor = getenv("RDP_COMM_FACTOR") ? atof(getenv("RDP_COMM_FACTOR")) : 1.2;

    MPI_CHECK(MPI_Comm_dup(_cartcomm, &cartcomm) );

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank) );
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    for(int i = 0; i < 27; ++i)
    {
	const int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };

	recv_tags[i] = (3 - d[0]) % 3 + 3 * ((3 - d[1]) % 3 + 3 * ((3 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, neighbor_ranks + i) );

	const int nhalodir[3] =  {
	    d[0] != 0 ? 1 : XSIZE_SUBDOMAIN,
	    d[1] != 0 ? 1 : YSIZE_SUBDOMAIN,
	    d[2] != 0 ? 1 : ZSIZE_SUBDOMAIN
	};

	const int nhalocells = nhalodir[0] * nhalodir[1] * nhalodir[2];

	const int estimate = numberdensity * safety_factor * nhalocells;

	CUDA_CHECK(cudaMalloc(&packbuffers[i].scattered_indices, sizeof(int) * estimate));

	if (i && estimate)
	{
	    CUDA_CHECK(cudaHostAlloc(&pinnedhost_sendbufs[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
	    CUDA_CHECK(cudaHostGetDevicePointer(&packbuffers[i].buffer, pinnedhost_sendbufs[i], 0));

	    CUDA_CHECK(cudaHostAlloc(&pinnedhost_recvbufs[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
	    CUDA_CHECK(cudaHostGetDevicePointer(&unpackbuffers[i].buffer, pinnedhost_recvbufs[i], 0));
	}
	else
	{
      	    CUDA_CHECK(cudaMalloc(&packbuffers[i].buffer, sizeof(float) * 6 * estimate));
	    unpackbuffers[i].buffer = packbuffers[i].buffer;

	    pinnedhost_sendbufs[i] = NULL;
	    pinnedhost_recvbufs[i] = NULL;
	}

	packbuffers[i].capacity = estimate;
	unpackbuffers[i].capacity = estimate;
	default_message_sizes[i] = estimate;
    }

    RedistributeParticlesKernels::texAllParticles.channelDesc = cudaCreateChannelDesc<float>();
    RedistributeParticlesKernels::texAllParticles.filterMode = cudaFilterModePoint;
    RedistributeParticlesKernels::texAllParticles.mipmapFilterMode = cudaFilterModePoint;
    RedistributeParticlesKernels::texAllParticles.normalized = 0;

    RedistributeParticlesKernels::texAllParticlesFloat2.channelDesc = cudaCreateChannelDesc<float2>();
    RedistributeParticlesKernels::texAllParticlesFloat2.filterMode = cudaFilterModePoint;
    RedistributeParticlesKernels::texAllParticlesFloat2.mipmapFilterMode = cudaFilterModePoint;
    RedistributeParticlesKernels::texAllParticlesFloat2.normalized = 0;

    CUDA_CHECK(cudaEventCreate(&evpacking, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&evsizes, cudaEventDisableTiming));
    //CUDA_CHECK(cudaEventCreate(&evcompaction, cudaEventDisableTiming));

CUDA_CHECK( cudaFuncSetCacheConfig( RedistributeParticlesKernels::gather_particles, cudaFuncCachePreferL1 ) );
}

void RedistributeParticles::_post_recv()
{
    for(int i = 1, c = 0; i < 27; ++i)
    	if (default_message_sizes[i])
	    MPI_CHECK( MPI_Irecv(recv_sizes + i, 1, MPI_INTEGER, neighbor_ranks[i], basetag + recv_tags[i], cartcomm, recvcountreq + c++) );
	else
	    recv_sizes[i] = 0;

    for(int i = 1, c = 0; i < 27; ++i)
	if (default_message_sizes[i])
	    MPI_CHECK( MPI_Irecv(pinnedhost_recvbufs[i], default_message_sizes[i] * 6, MPI_FLOAT,
				 neighbor_ranks[i], basetag + recv_tags[i] + 333, cartcomm, recvmsgreq + c++) );
}

void RedistributeParticles::_adjust_send_buffers(const int requested_capacities[27])
{
    for(int i = 0; i < 27; ++i)
    {
	if (requested_capacities[i] <= packbuffers[i].capacity)
	    continue;

	const int capacity = requested_capacities[i];

	CUDA_CHECK(cudaFree(packbuffers[i].scattered_indices));
	CUDA_CHECK(cudaMalloc(&packbuffers[i].scattered_indices, sizeof(int) * capacity));

	if (i)
	{
	    CUDA_CHECK(cudaFreeHost(pinnedhost_sendbufs[i]));

	    CUDA_CHECK(cudaHostAlloc(&pinnedhost_sendbufs[i], sizeof(float) * 6 * capacity, cudaHostAllocMapped));
	    CUDA_CHECK(cudaHostGetDevicePointer(&packbuffers[i].buffer, pinnedhost_sendbufs[i], 0));

	    packbuffers[i].capacity = capacity;
	}
	else
	{
	    CUDA_CHECK(cudaFree(packbuffers[i].buffer));

	    CUDA_CHECK(cudaMalloc(&packbuffers[i].buffer, sizeof(float) * 6 * capacity));
	    unpackbuffers[i].buffer = packbuffers[i].buffer;

	    assert(pinnedhost_sendbufs[i] == NULL);

	    packbuffers[i].capacity = capacity;
	    unpackbuffers[i].capacity = capacity;
	}
    }
}

bool RedistributeParticles::_adjust_recv_buffers(const int requested_capacities[27])
{
    bool haschanged = false;

    for(int i = 0; i < 27; ++i)
    {
	if (requested_capacities[i] <= unpackbuffers[i].capacity)
	    continue;

	haschanged = true;

	const int capacity = requested_capacities[i];

	if (i)
	{
	    //preserve-resize policy
	    float * const old = pinnedhost_recvbufs[i];

	    CUDA_CHECK(cudaHostAlloc(&pinnedhost_recvbufs[i], sizeof(float) * 6 * capacity, cudaHostAllocMapped));
	    CUDA_CHECK(cudaHostGetDevicePointer(&unpackbuffers[i].buffer, pinnedhost_recvbufs[i], 0));

	    CUDA_CHECK(cudaMemcpy(pinnedhost_recvbufs[i], old, sizeof(float) * 6 * unpackbuffers[i].capacity,
				  cudaMemcpyHostToHost));

	    CUDA_CHECK(cudaFreeHost(old));
	}
	else
	{
	    printf("RedistributeParticles::_adjust_recv_buffers i==0 ooooooooooooooops %d , req %d!!\n", unpackbuffers[i].capacity, capacity);
	    abort();
	    //CUDA_CHECK(cudaFree(unpackbuffers[i].buffer));
	    //CUDA_CHECK(cudaMalloc(&unpackbuffers[i].buffer, sizeof(float) * 6 * capacity));
	    //assert(pinnedhost_recvbufs[i] == NULL);
	}

	unpackbuffers[i].capacity = capacity;
    }

    return haschanged;
}

void RedistributeParticles::pack(const Particle * const particles, const int nparticles, cudaStream_t mystream)
{
    NVTX_RANGE("RDP/pack");

    bool secondchance = false;

    if (firstcall)
	_post_recv();

    size_t textureoffset;
    if (nparticles)
    CUDA_CHECK(cudaBindTexture(&textureoffset, &RedistributeParticlesKernels::texAllParticles, particles,
			       &RedistributeParticlesKernels::texAllParticles.channelDesc,
			       sizeof(float) * 6 * nparticles));

    if (nparticles)
    CUDA_CHECK(cudaBindTexture(&textureoffset, &RedistributeParticlesKernels::texAllParticlesFloat2, particles,
			       &RedistributeParticlesKernels::texAllParticlesFloat2.channelDesc,
			       sizeof(float) * 6 * nparticles));

    RedistributeParticlesKernels::ntexparticles = nparticles;
    RedistributeParticlesKernels::texparticledata = (float2 *)particles;

pack_attempt:

    CUDA_CHECK(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::pack_buffers, packbuffers,
					   sizeof(PackBuffer) * 27, 0, cudaMemcpyHostToDevice, mystream));

    *failure.data = false;
    RedistributeParticlesKernels::setup<<<1, 32, 0, mystream>>>();

    if (nparticles)
	RedistributeParticlesKernels::scatter_halo_indices_pack<<< (nparticles + 127) / 128, 128, 0, mystream>>>(nparticles);

    RedistributeParticlesKernels::tiny_scan<<<1, 32, 0, mystream>>>(nparticles, packbuffers[0].capacity, packsizes.devptr, failure.devptr);

    CUDA_CHECK(cudaEventRecord(evsizes, mystream));

#ifndef NDEBUG
    RedistributeParticlesKernels::check_scan<<<1, 1, 0, mystream>>>();
#endif

    if (nparticles)
	RedistributeParticlesKernels::pack<<< (3 * nparticles + 127) / 128, 128, 0, mystream>>> (nparticles, nparticles * 3);

    CUDA_CHECK(cudaEventRecord(evpacking, mystream));

    CUDA_CHECK(cudaEventSynchronize(evsizes));

    if (*failure.data)
    {
	//wait for packing to finish
	CUDA_CHECK(cudaEventSynchronize(evpacking));

	printf("RedistributeParticles::pack RANK %d ...FAILED! Recovering now...\n", myrank);

	_adjust_send_buffers(packsizes.data);

	if (myrank == 0)
	    for(int i = 0; i < 27; ++i)
		printf("ASD: %d\n", packsizes.data[i]);

	if (secondchance)
	{
	    printf("...non siamo qui a far la ceretta allo yeti.\n");
	    abort();
	}

	if (!secondchance)
	    secondchance = true;

	goto pack_attempt;
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

void RedistributeParticles::send()
{
    NVTX_RANGE("RDP/send", NVTX_C2);

    if (!firstcall)
	_waitall(sendcountreq, nactiveneighbors);

    for(int i = 0; i < 27; ++i)
	send_sizes[i] = packsizes.data[i];

    nbulk = recv_sizes[0] = send_sizes[0];

    {
	int c = 0;
	for(int i = 1; i < 27; ++i)
	    if (default_message_sizes[i])
		MPI_CHECK( MPI_Isend(send_sizes + i, 1, MPI_INTEGER, neighbor_ranks[i], basetag + i, cartcomm, sendcountreq + c++) );

	assert(c == nactiveneighbors);
    }

    CUDA_CHECK(cudaEventSynchronize(evpacking));

    if (!firstcall)
	_waitall(sendmsgreq, nsendmsgreq);

    nsendmsgreq = 0;
    for(int i = 1; i < 27; ++i)
	if (default_message_sizes[i])
	{
	    MPI_CHECK( MPI_Isend(pinnedhost_sendbufs[i], default_message_sizes[i] * 6, MPI_FLOAT, neighbor_ranks[i], basetag + i + 333,
				 cartcomm, sendmsgreq + nsendmsgreq) );

	    ++nsendmsgreq;
	}

    for(int i = 1; i < 27; ++i)
	if (default_message_sizes[i] && send_sizes[i] > default_message_sizes[i])
	{
	    const int count = send_sizes[i] - default_message_sizes[i];

	    MPI_CHECK( MPI_Isend(pinnedhost_sendbufs[i] + default_message_sizes[i] * 6, count * 6, MPI_FLOAT,
				 neighbor_ranks[i], basetag + i + 666, cartcomm, sendmsgreq + nsendmsgreq) );
	    ++nsendmsgreq;
	}

    assert(nactiveneighbors <= nsendmsgreq && nsendmsgreq <= 2 * nactiveneighbors);
}

void RedistributeParticles::bulk(const int nparticles, int * const cellstarts, int * const cellcounts, cudaStream_t mystream)
{
    CUDA_CHECK(cudaMemsetAsync(cellcounts, 0, sizeof(int) * XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN, mystream));
/*    CUDA_CHECK(cudaPeekAtLastError());
    dim3 bs(8, 8, 8);

    dim3 gs((XSIZE_SUBDOMAIN + bs.x - 1) / bs.x,
	    (YSIZE_SUBDOMAIN + bs.y - 1) / bs.y,
	    (ZSIZE_SUBDOMAIN + bs.z - 1) / bs.z);

    subindices.resize(nparticles);
*/
    subindices.resize(nparticles);

    if (nparticles)
    subindex_local<<< (nparticles + 127) / 128, 128, 0, mystream>>>
	(nparticles, RedistributeParticlesKernels::texparticledata, cellcounts, subindices.data);
/*
#ifndef NDEBUG
    CUDA_CHECK(cudaDeviceSynchronize());

    {
	const int n =  XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;
	int * c = new int[n];
	cudaMemcpy(c, cellcounts, sizeof(int) * n, cudaMemcpyDeviceToHost);
	for(int i = 0; i < n; ++i)
	    assert(c[i] == 4);
	delete [] c;

	int * w = new unit4[n];
	cudaMemcpy(c, cellcounts, sizeof(int) * n, cudaMemcpyDeviceToHost);
	for(int i = 0; i < n; ++i)
	    assert(c[i] == 4);
	delete [] c;

    }
    #endif*/
    //RedistributeParticlesKernels::subindex_local<0><<<gs, bs, 0, mystream>>>(nparticles, cellstarts, cellcounts, subindices.data);
    //RedistributeParticlesKernels::subindex_local<1><<<gs, bs, 0, mystream>>>(nparticles, cellstarts, cellcounts, subindices.data);

    CUDA_CHECK(cudaPeekAtLastError());
}

int RedistributeParticles::recv_count(cudaStream_t mystream, float& host_idle_time)
{
    CUDA_CHECK(cudaPeekAtLastError());

    NVTX_RANGE("RDP/recv-count", NVTX_C3);

    host_idle_time += _waitall(recvcountreq, nactiveneighbors);

    {
	static int usize[27], ustart[28], ustart_padded[28];

	usize[0] = 0;
	for(int i = 1; i < 27; ++i)
	    usize[i] = recv_sizes[i] * (default_message_sizes[i] > 0);

	ustart[0] = 0;
	for(int i = 1; i < 28; ++i)
	    ustart[i] = ustart[i - 1] + usize[i - 1];

	nexpected = nbulk + ustart[27];
	nhalo = ustart[27];

	ustart_padded[0] = 0;
	for(int i = 1; i < 28; ++i)
	    ustart_padded[i] = ustart_padded[i - 1] + 32 * ((usize[i - 1] + 31) / 32);

	nhalo_padded = ustart_padded[27];

	CUDA_CHECK(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::unpack_start, ustart,
					   sizeof(int) * 28, 0, cudaMemcpyHostToDevice, mystream));

	CUDA_CHECK(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::unpack_start_padded, ustart_padded,
					   sizeof(int) * 28, 0, cudaMemcpyHostToDevice, mystream));
    }

    {
	remote_particles.resize(nhalo);
	subindices_remote.resize(nhalo);
	scattered_indices.resize(nexpected);
    }

    firstcall = false;

    return nexpected;
}

void RedistributeParticles::recv_unpack(Particle * const particles, float4 * const xyzouvwo, ushort4 * const xyzo_half, const int nparticles,
					int * const cellstarts, int * const cellcounts, cudaStream_t mystream, float& host_idling_time)
{
    NVTX_RANGE("RDP/recv-unpack", NVTX_C4);

    assert(nparticles == nexpected);

    host_idling_time += _waitall(recvmsgreq, nactiveneighbors);

    const bool haschanged = true;
    _adjust_recv_buffers(recv_sizes);

    if (haschanged)
	CUDA_CHECK(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::unpack_buffers, unpackbuffers,
					       sizeof(UnpackBuffer) * 27, 0, cudaMemcpyHostToDevice, mystream));

    for(int i = 1; i < 27; ++i)
	if (default_message_sizes[i] && recv_sizes[i] > default_message_sizes[i])
	{
	    const int count = recv_sizes[i] - default_message_sizes[i];

	    MPI_Status status;
	    MPI_CHECK( MPI_Recv(pinnedhost_recvbufs[i] + default_message_sizes[i] * 6, count * 6, MPI_FLOAT,
				neighbor_ranks[i], basetag + recv_tags[i] + 666, cartcomm, &status) );
	}

    CUDA_CHECK(cudaPeekAtLastError());

#ifndef NDEBUG
    CUDA_CHECK(cudaMemset(remote_particles.data, 0xff, sizeof(Particle) * remote_particles.size));
#endif

    if (nhalo)
	RedistributeParticlesKernels::subindex_remote<<< (nhalo_padded + 127) / 128, 128, 0, mystream >>>
	    (nhalo_padded, nhalo, cellcounts, (float2 *)remote_particles.data, subindices_remote.data);

    if (compressed_cellcounts.size)
    compress_counts<<< (compressed_cellcounts.size + 127) / 128, 128, 0, mystream >>>
	(compressed_cellcounts.size, (int4 *)cellcounts, (uchar4 *)compressed_cellcounts.data);

    scan(compressed_cellcounts.data, compressed_cellcounts.size, mystream, (uint *)cellstarts);

#ifndef NDEBUG
    CUDA_CHECK(cudaMemset(scattered_indices.data, 0xff, sizeof(int) * scattered_indices.size));
#endif

    if (subindices.size)
    RedistributeParticlesKernels::scatter_indices<<< (subindices.size + 127) / 128, 128, 0, mystream>>>
	(false, subindices.data, subindices.size, cellstarts, scattered_indices.data, scattered_indices.size);

    if (nhalo)
	RedistributeParticlesKernels::scatter_indices<<< (nhalo + 127) / 128, 128, 0, mystream>>>
	    (true, subindices_remote.data, nhalo, cellstarts, scattered_indices.data, scattered_indices.size);

    assert(scattered_indices.size == nparticles);

    if (nparticles)
    RedistributeParticlesKernels::gather_particles<<< (nparticles + 127) / 128, 128, 0, mystream>>>
	(scattered_indices.data, (float2 *)remote_particles.data, nhalo,
	 RedistributeParticlesKernels::ntexparticles, nparticles, (float2 *)particles, xyzouvwo, xyzo_half);

    CUDA_CHECK(cudaPeekAtLastError());

#ifndef NDEBUG
    RedistributeParticlesKernels::check<<<(XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN + 127) / 128, 128, 0, mystream>>>(cellstarts, cellcounts, particles, nparticles);
#endif

    _post_recv();

    CUDA_CHECK(cudaPeekAtLastError());
}

void RedistributeParticles::_cancel_recv()
{
    if (!firstcall)
    {
	_waitall(sendcountreq, nactiveneighbors);
	_waitall(sendmsgreq, nsendmsgreq);

	for(int i = 0; i < nactiveneighbors; ++i)
	    MPI_CHECK( MPI_Cancel(recvcountreq + i) );

	for(int i = 0; i < nactiveneighbors; ++i)
	    MPI_CHECK( MPI_Cancel(recvmsgreq + i) );

	firstcall = true;
    }
}

void RedistributeParticles::adjust_message_sizes(ExpectedMessageSizes sizes)
{
    _cancel_recv();

    nactiveneighbors = 0;
    for(int i = 1; i < 27; ++i)
    {
	const int d[3] = { (i + 1) % 3, (i / 3 + 1) % 3, (i / 9 + 1) % 3 };
       	const int entry = d[0] + 3 * (d[1] + 3 * d[2]);

	int estimate = (int)ceil(safety_factor * sizes.msgsizes[entry]);
	estimate = 32 * ((estimate + 31) / 32);

	default_message_sizes[i] = estimate;
	nactiveneighbors += (estimate > 0);
    }

    _adjust_send_buffers(default_message_sizes);
    _adjust_recv_buffers(default_message_sizes);
}

RedistributeParticles::~RedistributeParticles()
{
    CUDA_CHECK(cudaEventDestroy(evpacking));
    CUDA_CHECK(cudaEventDestroy(evsizes));

    _cancel_recv();

    for(int i = 0; i < 27; ++i)
    {
	CUDA_CHECK(cudaFree(packbuffers[i].scattered_indices));

	if (i)
	    CUDA_CHECK(cudaFreeHost(packbuffers[i].buffer));
	else
	    CUDA_CHECK(cudaFree(packbuffers[i].buffer));
    }
}
