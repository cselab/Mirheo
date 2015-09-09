#pragma once

#include "common.h"

__device__ __forceinline__
void read_AOS6f(const float2 * const data, const int nparticles, float2& s0, float2& s1, float2& s2)
{
    if (nparticles == 0)
	return;

    int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

    const int nfloat2 = 3 * nparticles;

    if (laneid < nfloat2)
	s0 = data[laneid];

    if (laneid + 32 < nfloat2)
	s1 = data[laneid + 32];

    if (laneid + 64 < nfloat2)
	s2 = data[laneid + 64];

    const int srclane0 = (3 * laneid + 0) & 0x1f;
    const int srclane1 = (srclane0 + 1) & 0x1f;
    const int srclane2 = (srclane0 + 2) & 0x1f;

    const int start = laneid % 3;

    {
	const float t0 = __shfl(start == 0 ? s0.x : start == 1 ? s1.x : s2.x, srclane0);
	const float t1 = __shfl(start == 0 ? s2.x : start == 1 ? s0.x : s1.x, srclane1);
	const float t2 = __shfl(start == 0 ? s1.x : start == 1 ? s2.x : s0.x, srclane2);

	s0.x = t0;
	s1.x = t1;
	s2.x = t2;
    }

    {
	const float t0 = __shfl(start == 0 ? s0.y : start == 1 ? s1.y : s2.y, srclane0);
	const float t1 = __shfl(start == 0 ? s2.y : start == 1 ? s0.y : s1.y, srclane1);
	const float t2 = __shfl(start == 0 ? s1.y : start == 1 ? s2.y : s0.y, srclane2);

	s0.y = t0;
	s1.y = t1;
	s2.y = t2;
    }
}

__device__ __forceinline__
void write_AOS6f(float2 * const data, const int nparticles, float2& s0, float2& s1, float2& s2)
{
    if (nparticles == 0)
	return;

    int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

    const int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
    const int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
    const int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

    const int start = laneid % 3;

    {
	const float t0 = __shfl(s0.x, srclane0);
	const float t1 = __shfl(s2.x, srclane1);
	const float t2 = __shfl(s1.x, srclane2);

	s0.x = start == 0 ? t0 : start == 1 ? t2 : t1;
	s1.x = start == 0 ? t1 : start == 1 ? t0 : t2;
	s2.x = start == 0 ? t2 : start == 1 ? t1 : t0;
    }

    {
	const float t0 = __shfl(s0.y, srclane0);
	const float t1 = __shfl(s2.y, srclane1);
	const float t2 = __shfl(s1.y, srclane2);

	s0.y = start == 0 ? t0 : start == 1 ? t2 : t1;
	s1.y = start == 0 ? t1 : start == 1 ? t0 : t2;
	s2.y = start == 0 ? t2 : start == 1 ? t1 : t0;
    }

    const int nfloat2 = 3 * nparticles;

    if (laneid < nfloat2)
	data[laneid] = s0;

    if (laneid + 32 < nfloat2)
	data[laneid + 32] = s1;

    if (laneid + 64 < nfloat2)
	data[laneid + 64] = s2;
}

__device__ __forceinline__
void read_AOS3f(const float * const data, const int nparticles, float& s0, float& s1, float& s2)
{
    if (nparticles == 0)
	return;

    int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

    const int nfloat2 = 3 * nparticles;

    if (laneid < nfloat2)
	s0 = data[laneid];

    if (laneid + 32 < nfloat2)
	s1 = data[laneid + 32];

    if (laneid + 64 < nfloat2)
	s2 = data[laneid + 64];

    const int srclane0 = (3 * laneid + 0) & 0x1f;
    const int srclane1 = (srclane0 + 1) & 0x1f;
    const int srclane2 = (srclane0 + 2) & 0x1f;

    const int start = laneid % 3;

    {
	const float t0 = __shfl(start == 0 ? s0 : start == 1 ? s1 : s2, srclane0);
	const float t1 = __shfl(start == 0 ? s2 : start == 1 ? s0 : s1, srclane1);
	const float t2 = __shfl(start == 0 ? s1 : start == 1 ? s2 : s0, srclane2);

	s0 = t0;
	s1 = t1;
	s2 = t2;
    }
}


__device__ __forceinline__
void write_AOS3f(float * const data, const int nparticles, float& s0, float& s1, float& s2)
{
    if (nparticles == 0)
	return;

    int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

    const int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
    const int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
    const int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

    const int start = laneid % 3;

    {
	const float t0 = __shfl(s0, srclane0);
	const float t1 = __shfl(s2, srclane1);
	const float t2 = __shfl(s1, srclane2);

	s0 = start == 0 ? t0 : start == 1 ? t2 : t1;
	s1 = start == 0 ? t1 : start == 1 ? t0 : t2;
	s2 = start == 0 ? t2 : start == 1 ? t1 : t0;
    }

    const int nfloat = 3 * nparticles;

    if (laneid < nfloat)
	data[laneid] = s0;

    if (laneid + 32 < nfloat)
	data[laneid + 32] = s1;

    if (laneid + 64 < nfloat)
	data[laneid + 64] = s2;
}

__global__ static void subindex_local(const int nparticles, const float2 * particles, int * const partials,
				      uchar4 * const subindices)
{
    assert(blockDim.x == 128 && blockDim.x * gridDim.x >= nparticles);

    const int lane = threadIdx.x & 0x1f;
    const int warpid = threadIdx.x >> 5;
    const int base = 32 * (warpid + 4 * blockIdx.x);
    const int nsrc = min(32, nparticles - base);

    if (nsrc == 0)
	return;

    int cid = -1;

    //LOAD PARTICLES, COMPUTE CELL ID
    {
	float2 data0, data1, data2;

	read_AOS6f(particles + 3 * base, nsrc, data0, data1, data2);

	const bool inside = (data0.x >= -XSIZE_SUBDOMAIN / 2 && data0.x < XSIZE_SUBDOMAIN / 2 &&
			     data0.y >= -YSIZE_SUBDOMAIN / 2 && data0.y < YSIZE_SUBDOMAIN / 2 &&
			     data1.x >= -ZSIZE_SUBDOMAIN / 2 && data1.x < ZSIZE_SUBDOMAIN / 2 );

	if (lane < nsrc && inside)
	{
	    const int xcid = (int)floor((double)data0.x + XSIZE_SUBDOMAIN / 2);
	    const int ycid = (int)floor((double)data0.y + YSIZE_SUBDOMAIN / 2);
	    const int zcid = (int)floor((double)data1.x + ZSIZE_SUBDOMAIN / 2);

	    cid = xcid + XSIZE_SUBDOMAIN * (ycid + YSIZE_SUBDOMAIN * zcid);
	}
    }

    int pid = lane + base;

    //BITONIC SORT
    {
#pragma unroll
	for(int D = 1; D <= 16; D <<= 1)
#pragma unroll
	    for(int L = D; L >= 1; L >>= 1)
	    {
		const int mask = L == D ? 2 * D - 1 : L;

		const int othercid = __shfl_xor(cid, mask);
		const int otherpid = __shfl_xor(pid, mask);

		const bool exchange =  (2 * (int)(lane < (lane ^ mask)) - 1) * (cid - othercid) > 0;

		if (exchange)
		{
		    cid = othercid;
		    pid = otherpid;
		}
	    }
    }

    int start, pcount;

    //FIND PARTICLES SHARING SAME CELL IDS
    {
	__shared__ volatile int keys[4][32];

	keys[warpid][lane] = cid;

	const bool ishead = cid != __shfl(cid, lane - 1) || lane == 0;

	if (cid >= 0)
	{
	    const int searchval = ishead ? cid + 1 : cid;

	    int first = ishead ? lane : 0;
	    int last = ishead ? 32 : (lane + 1);
	    int count = last - first;

	    while (count > 0)
	    {
		const int step = count / 2;
		const int it = first + step;

		if (keys[warpid][it] < searchval)
		{
		    first = it + 1;
		    count -= step + 1;
		}
		else
		    count = step;
	    }

	    start = ishead ? lane : first;

	    if (ishead)
		pcount = first - lane;
	}
    }

    //ADD COUNT TO PARTIALS, WRITE SUBINDEX
    {
	int globalstart;

	if (cid >= 0 && lane == start)
	    globalstart = atomicAdd(partials + cid, pcount);

	const int subindex = __shfl(globalstart, start) + (lane - start);
	assert(subindex < 0xff && subindex >= 0 || cid < 0 );

	uchar4 entry = make_uchar4(0xff, 0xff, 0xff, 0xff);

	if (cid >= 0)
	{
	    const int xcid = cid % XSIZE_SUBDOMAIN;
	    const int ycid = (cid / XSIZE_SUBDOMAIN) % YSIZE_SUBDOMAIN;
	    const int zcid = cid / (XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN);

	    entry = make_uchar4(xcid, ycid, zcid, subindex);
	}

	if (pid < nparticles)
	    subindices[pid] = entry;
    }
}

__global__ static void compress_counts(const int nentries, const int4 * const counts, uchar4 * const output)
{
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (4 * gid >= nentries)
	return;

    assert(nentries % 4 == 0);

    const int4 entry = counts[gid];

    output[gid] = make_uchar4(entry.x, entry.y, entry.z, entry.w);
}
