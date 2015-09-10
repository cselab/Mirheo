/*
 * contact.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

static const int maxsolutes = 32;

#include <../dpd-rng.h>

#include "common-kernels.h"
#include "scan.h"
#include "contact.h"

namespace KernelsContact
{
    enum
    {
	XCELLS = XSIZE_SUBDOMAIN,
	YCELLS = YSIZE_SUBDOMAIN,
	ZCELLS = ZSIZE_SUBDOMAIN,
	XOFFSET = XCELLS / 2,
	YOFFSET = YCELLS / 2,
	ZOFFSET = ZCELLS / 2
    };

    static const int NCELLS = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;

    union CellEntry { int pid; uchar4 code; };

    struct Params { float gamma, sigmaf, rc2; };

    __constant__ Params params;

    texture<int, cudaTextureType1D> texCellsStart, texCellEntries;

    __global__ void bulk_3tpp(const float2 * const particles, const int np, const int ncellentries, const int nsolutes,
			      float * const acc, const float seed, const int mysoluteid);

    __global__ 	void halo(const int nparticles_padded, const int ncellentries, const int nsolutes, const float seed);

    void setup()
    {
	texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
	texCellsStart.filterMode = cudaFilterModePoint;
	texCellsStart.mipmapFilterMode = cudaFilterModePoint;
	texCellsStart.normalized = 0;

	texCellEntries.channelDesc = cudaCreateChannelDesc<int>();
	texCellEntries.filterMode = cudaFilterModePoint;
	texCellEntries.mipmapFilterMode = cudaFilterModePoint;
	texCellEntries.normalized = 0;

	CUDA_CHECK(cudaFuncSetCacheConfig(bulk_3tpp, cudaFuncCachePreferL1));
	CUDA_CHECK(cudaFuncSetCacheConfig(halo, cudaFuncCachePreferL1));
    }
}

ComputeContact::ComputeContact(MPI_Comm comm):
cellsstart(KernelsContact::NCELLS + 16), cellscount(KernelsContact::NCELLS + 16), compressed_cellscount(KernelsContact::NCELLS + 16)
{
    int myrank;
    MPI_CHECK( MPI_Comm_rank(comm, &myrank));

    local_trunk = Logistic::KISS(7119 - myrank, 187 + myrank, 18278, 15674);

    KernelsContact::Params params = { gammadpd, sigmaf, 1};

    CUDA_CHECK(cudaMemcpyToSymbol(KernelsContact::params, &params, sizeof(params)));

    CUDA_CHECK(cudaPeekAtLastError());
}

namespace KernelsContact
{
    __global__ void populate(const uchar4 * const subindices, const int * const cellstart,
			     const int nparticles, const int soluteid, const int ntotalparticles,
			     CellEntry * const entrycells)
    {
#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined! assuming 350
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

	assert(blockDim.x == 128);

	const int warpid = threadIdx.x >> 5;
	const int tid = threadIdx.x & 0x1f;

	const int base = 32 * (warpid + 4 * blockIdx.x);
	const int pid = base + tid;

	if (pid >= nparticles)
	    return;

	const uchar4 subindex = subindices[pid];

	if (subindex.x == 0xff && subindex.y == 0xff && subindex.z == 0xff)
	    return;

	assert(subindex.x < XCELLS && subindex.y < YCELLS && subindex.z < ZCELLS);

	const int cellid = subindex.x + XCELLS * (subindex.y + YCELLS * subindex.z);
	const int mystart = _ACCESS(cellstart + cellid);
	const int slot = mystart + subindex.w;
	assert(slot < ntotalparticles);

	CellEntry myentrycell;
	myentrycell.pid = pid;
	myentrycell.code.w = soluteid;

	entrycells[slot] = myentrycell;
    }

    __constant__ int cnsolutes[maxsolutes];
    __constant__ const float2 * csolutes[maxsolutes];
    __constant__ float * csolutesacc[maxsolutes];

    void bind(const int * const cellsstart, const int * const cellentries, const int ncellentries,
	      std::vector<ParticlesWrap> wsolutes, cudaStream_t stream, const int * const cellscount)
    {
	size_t textureoffset = 0;

	if (ncellentries)
	    CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellEntries, cellentries, &texCellEntries.channelDesc,
				       sizeof(int) * ncellentries));

	assert(textureoffset == 0);

	const int ncells = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;

	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart, &texCellsStart.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);

	const int n = wsolutes.size();

	int ns[n];
	float2 * ps[n];
	float * as[n];

	for(int i = 0; i < n; ++i)
	{
	    ns[i] = wsolutes[i].n;
	    ps[i] = (float2 *)wsolutes[i].p;
	    as[i] = (float * )wsolutes[i].a;
	}

	CUDA_CHECK(cudaMemcpyToSymbolAsync(cnsolutes, ns, sizeof(int) * n, 0, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(csolutes, ps, sizeof(float2 *) * n, 0, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(csolutesacc, as, sizeof(float *) * n, 0, cudaMemcpyHostToDevice, stream));
    }
}

void ComputeContact::build_cells(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream)
{
    this->nsolutes = wsolutes.size();

    int ntotal = 0;

    for(int i = 0; i < wsolutes.size(); ++i)
	ntotal += wsolutes[i].n;

    subindices.resize(ntotal);
    cellsentries.resize(ntotal);

    CUDA_CHECK(cudaMemsetAsync(cellscount.data, 0, sizeof(int) * cellscount.size, stream));
    
#ifndef NDEBUG
    CUDA_CHECK(cudaMemsetAsync(cellsentries.data, 0xff, sizeof(int) * cellsentries.capacity, stream));
    CUDA_CHECK(cudaMemsetAsync(subindices.data, 0xff, sizeof(int) * subindices.capacity, stream));
    CUDA_CHECK(cudaMemsetAsync(compressed_cellscount.data, 0xff, sizeof(unsigned char) * compressed_cellscount.capacity, stream));
    CUDA_CHECK(cudaMemsetAsync(cellsstart.data, 0xff, sizeof(int) * cellsstart.capacity, stream));
#endif

    CUDA_CHECK(cudaPeekAtLastError());

    int ctr = 0;
    for(int i = 0; i < wsolutes.size(); ++i)
    {
	const ParticlesWrap it = wsolutes[i];

	if (it.n)
	    subindex_local<<< (it.n + 127) / 128, 128, 0, stream >>>(it.n, (float2 *)it.p, cellscount.data, subindices.data + ctr);

	ctr += it.n;
    }

    compress_counts<<< (compressed_cellscount.size + 127) / 128, 128, 0, stream >>>
	(compressed_cellscount.size, (int4 *)cellscount.data, (uchar4 *)compressed_cellscount.data);

    scan(compressed_cellscount.data, compressed_cellscount.size, stream, (uint *)cellsstart.data);

    ctr = 0;
    for(int i = 0; i < wsolutes.size(); ++i)
    {
	const ParticlesWrap it = wsolutes[i];

	if (it.n)
	    KernelsContact::populate<<< (it.n + 127) / 128, 128, 0, stream >>>
		(subindices.data + ctr, cellsstart.data, it.n, i, ntotal, (KernelsContact::CellEntry *)cellsentries.data);

	ctr += it.n;
    }

    CUDA_CHECK(cudaPeekAtLastError());

    KernelsContact::bind(cellsstart.data, cellsentries.data, ntotal, wsolutes, stream, cellscount.data);
}

namespace KernelsContact
{
    __global__ void bulk_3tpp(const float2 * const particles,
			      const int np, const int ncellentries, const int nsolutes,
			      float * const acc, const float seed, const int mysoluteid)
    {
	assert(blockDim.x * gridDim.x >= np * 3);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
       	const int pid = gid / 3;
	const int zplane = gid % 3;

	if (pid >= np)
	    return;

	const float2 dst0 = _ACCESS(particles + 3 * pid + 0);
	const float2 dst1 = _ACCESS(particles + 3 * pid + 1);
	const float2 dst2 = _ACCESS(particles + 3 * pid + 2);

	int scan1, scan2, ncandidates, spidbase;
	int deltaspid1, deltaspid2;
	bool isinside = false;

	{
	    const int xcenter = XOFFSET + (int)floorf(dst0.x);
	    const int xstart = max(0, xcenter - 1);
	    const int xcount = min(XCELLS, xcenter + 2) - xstart;

	    if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0)
		return;

	    assert(xcount >= 0);

	    const int ycenter = YOFFSET + (int)floorf(dst0.y);

	    const int zcenter = ZOFFSET + (int)floorf(dst1.x);
	    const int zmy = zcenter - 1 + zplane;
	    const bool zvalid = zmy >= 0 && zmy < ZCELLS;

	    int count0 = 0, count1 = 0, count2 = 0;

	    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS)
	    {
		const int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
		assert(cid0 >= 0 && cid0 + xcount <= NCELLS);
		spidbase = tex1Dfetch(texCellsStart, cid0);
		count0 = tex1Dfetch(texCellsStart, cid0 + xcount) - spidbase;
	    }

	    if (zvalid && ycenter >= 0 && ycenter < YCELLS)
	    {
		const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
		assert(cid1 >= 0 && cid1 + xcount <= NCELLS);
		deltaspid1 = tex1Dfetch(texCellsStart, cid1);
		count1 = tex1Dfetch(texCellsStart, cid1 + xcount) - deltaspid1;

		isinside = xcenter >= 0 && xcenter < XCELLS;
	    }

	    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS)
	    {
		const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
		deltaspid2 = tex1Dfetch(texCellsStart, cid2);
		assert(cid2 >= 0 && cid2 + xcount <= NCELLS);
		count2 = tex1Dfetch(texCellsStart, cid2 + xcount) - deltaspid2;
	    }

	    scan1 = count0;
	    scan2 = count0 + count1;
	    ncandidates = scan2 + count2;

	    deltaspid1 -= scan1;
	    deltaspid2 -= scan2;
	}

	float xforce = 0, yforce = 0, zforce = 0;

#pragma unroll 3
	for(int i = 0; i < ncandidates; ++i)
	{
	    const int m1 = (int)(i >= scan1);
	    const int m2 = (int)(i >= scan2);
	    const int slot = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);
	    assert(slot >= 0 && slot < ncellentries);

	    CellEntry ce;
	    ce.pid = tex1Dfetch(texCellEntries, slot);
	    const int soluteid = ce.code.w;

	    assert(soluteid >= 0 && soluteid < nsolutes);
	    ce.code.w = 0;

	    const int spid = ce.pid;
	    assert(spid >= 0 && spid < cnsolutes[soluteid]);

	    if (isinside && (mysoluteid < soluteid || mysoluteid == soluteid && pid <= spid))
		continue;

	    const int sentry = 3 * spid;
	    const float2 stmp0 = _ACCESS(csolutes[soluteid] +  sentry    );
	    const float2 stmp1 = _ACCESS(csolutes[soluteid] +  sentry + 1);
	    const float2 stmp2 = _ACCESS(csolutes[soluteid] +  sentry + 2);

	    const float _xr = dst0.x - stmp0.x;
	    const float _yr = dst0.y - stmp0.y;
	    const float _zr = dst1.x - stmp1.x;

	    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	    assert(rij2 > 0);

	    const float invrij = rsqrtf(rij2);

	    const float rij = rij2 * invrij;

	    if (rij2 >= params.rc2)
		continue;

	    const float invr2 = invrij * invrij;
	    const float t2 = 0.0625f * invr2;
	    const float t4 = t2 * t2;
	    const float t6 = t4 * t2;
	    const float lj = max(0.f, -24.f * invr2 * t6 * (2.f * t6 - 1.f));

	    const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(1.f - rij);

	    const float xr = _xr * invrij;
	    const float yr = _yr * invrij;
	    const float zr = _zr * invrij;

	    const float rdotv =
		xr * (dst1.y - stmp1.y) +
		yr * (dst2.x - stmp2.x) +
		zr * (dst2.y - stmp2.y);

	    const float myrandnr = Logistic::mean0var1(seed, pid, spid);

	    const float strength = lj + (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

	    const float xinteraction = strength * xr;
	    const float yinteraction = strength * yr;
	    const float zinteraction = strength * zr;

	    xforce += xinteraction;
	    yforce += yinteraction;
	    zforce += zinteraction;

	    assert(!isnan(xinteraction));
	    assert(!isnan(yinteraction));
	    assert(!isnan(zinteraction));

	    assert(fabs(xinteraction) < 1e4);
	    assert(fabs(yinteraction) < 1e4);
	    assert(fabs(zinteraction) < 1e4);

	    atomicAdd(csolutesacc[soluteid] + sentry    , -xinteraction);
	    atomicAdd(csolutesacc[soluteid] + sentry + 1, -yinteraction);
	    atomicAdd(csolutesacc[soluteid] + sentry + 2, -zinteraction);
	}

	atomicAdd(acc + 3 * pid + 0, xforce);
	atomicAdd(acc + 3 * pid + 1, yforce);
	atomicAdd(acc + 3 * pid + 2, zforce);

	for(int c = 0; c < 3; ++c)
	    assert(!isnan(acc[3 * pid + c]));
    }
}

void ComputeContact::bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream)
{
    NVTX_RANGE("Contact/bulk", NVTX_C6);

    if (wsolutes.size() == 0)
	return;

    for(int i = 0; i < wsolutes.size(); ++i)
    {
	ParticlesWrap it = wsolutes[i];

   	if (it.n)
	    KernelsContact::bulk_3tpp<<< (3 * it.n + 127) / 128, 128, 0, stream >>>
		((float2 *)it.p, it.n, cellsentries.size, wsolutes.size(), (float *)it.a, local_trunk.get_float(), i);

	CUDA_CHECK(cudaPeekAtLastError());
    }
}

namespace KernelsContact
{
    __constant__ int packstarts_padded[27], packcount[26];
    __constant__ Particle * packstates[26];
    __constant__ Acceleration * packresults[26];

    __global__ 	void halo(const int nparticles_padded, const int ncellentries, const int nsolutes, const float seed)
    {
	assert(blockDim.x * gridDim.x >= nparticles_padded);

	const int laneid = threadIdx.x & 0x1f;
	const int warpid = threadIdx.x >> 5;
	const int localbase = 32 * (warpid + 4 * blockIdx.x);
	const int pid = localbase + laneid;

	if (localbase >= nparticles_padded)
	    return;

	int nunpack;
	float2 dst0, dst1, dst2;
	float * dst = NULL;

	{
	    const uint key9 = 9 * (localbase >= packstarts_padded[9]) + 9 * (localbase >= packstarts_padded[18]);
	    const uint key3 = 3 * (localbase >= packstarts_padded[key9 + 3]) + 3 * (localbase >= packstarts_padded[key9 + 6]);
	    const uint key1 = (localbase >= packstarts_padded[key9 + key3 + 1]) + (localbase >= packstarts_padded[key9 + key3 + 2]);
	    const int code = key9 + key3 + key1;
	    assert(code >= 0 && code < 26);
	    assert(localbase >= packstarts_padded[code] && localbase < packstarts_padded[code + 1]);

	    const int unpackbase = localbase - packstarts_padded[code];
	    assert (unpackbase >= 0);
	    assert(unpackbase < packcount[code]);

	    nunpack = min(32, packcount[code] - unpackbase);

	    if (nunpack == 0)
		return;

	    read_AOS6f((float2 *)(packstates[code] + unpackbase), nunpack, dst0, dst1, dst2);

	    dst = (float*)(packresults[code] + unpackbase);
	}

	float xforce, yforce, zforce;
	read_AOS3f(dst, nunpack, xforce, yforce, zforce);

	const int nzplanes = laneid < nunpack ? 3 : 0;

	for(int zplane = 0; zplane < nzplanes; ++zplane)
	{
	    int scan1, scan2, ncandidates, spidbase;
	    int deltaspid1, deltaspid2;

	    {
       		const int xcenter = XOFFSET + (int)floorf(dst0.x);
		const int xstart = max(0, xcenter - 1);
		const int xcount = min(XCELLS, xcenter + 2) - xstart;

		if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0)
		    continue;

		assert(xcount >= 0);

		const int ycenter = YOFFSET + (int)floorf(dst0.y);

		const int zcenter = ZOFFSET + (int)floorf(dst1.x);
		const int zmy = zcenter - 1 + zplane;
		const bool zvalid = zmy >= 0 && zmy < ZCELLS;

		int count0 = 0, count1 = 0, count2 = 0;

		if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS)
		{
		    const int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
		    assert(cid0 >= 0 && cid0 + xcount <= NCELLS);
		    spidbase = tex1Dfetch(texCellsStart, cid0);
		    count0 = tex1Dfetch(texCellsStart, cid0 + xcount) - spidbase;
		}

		if (zvalid && ycenter >= 0 && ycenter < YCELLS)
		{
		    const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
		    assert(cid1 >= 0 && cid1 + xcount <= NCELLS);
		    deltaspid1 = tex1Dfetch(texCellsStart, cid1);
		    count1 = tex1Dfetch(texCellsStart, cid1 + xcount) - deltaspid1;
		}

		if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS)
		{
		    const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
		    deltaspid2 = tex1Dfetch(texCellsStart, cid2);
		    assert(cid2 >= 0 && cid2 + xcount <= NCELLS);
		    count2 = tex1Dfetch(texCellsStart, cid2 + xcount) - deltaspid2;
		}

		scan1 = count0;
		scan2 = count0 + count1;
		ncandidates = scan2 + count2;

		deltaspid1 -= scan1;
		deltaspid2 -= scan2;
	    }

	    for(int i = 0; i < ncandidates; ++i)
	    {
		const int m1 = (int)(i >= scan1);
		const int m2 = (int)(i >= scan2);
		const int slot = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

		assert(slot >= 0 && slot < ncellentries);
		CellEntry ce;
		ce.pid = tex1Dfetch(texCellEntries, slot);
		const int soluteid = ce.code.w;
		assert(soluteid >= 0 && soluteid < nsolutes);
		ce.code.w = 0;

		const int spid = ce.pid;
		assert(spid >= 0 && spid < cnsolutes[soluteid]);

		const int sentry = 3 * spid;
		const float2 stmp0 = _ACCESS(csolutes[soluteid] + sentry    );
		const float2 stmp1 = _ACCESS(csolutes[soluteid] + sentry + 1);
		const float2 stmp2 = _ACCESS(csolutes[soluteid] + sentry + 2);

		const float _xr = dst0.x - stmp0.x;
		const float _yr = dst0.y - stmp0.y;
		const float _zr = dst1.x - stmp1.x;

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		assert(rij2 > 0);

		const float invrij = rsqrtf(rij2);

		const float rij = rij2 * invrij;

		if (rij2 >= params.rc2)
		    continue;

		const float invr2 = invrij * invrij;
		const float t2 = 0.0625f * invr2;
		const float t4 = t2 * t2;
		const float t6 = t4 * t2;
		const float lj = max(0.f, -24.f * invr2 * t6 * (2.f * t6 - 1.f));

		const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(1.f - rij);

		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv =
		    xr * (dst1.y - stmp1.y) +
		    yr * (dst2.x - stmp2.x) +
		    zr * (dst2.y - stmp2.y);

		const float myrandnr = Logistic::mean0var1(seed, pid, spid);

		const float strength = lj + (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

		const float xinteraction = strength * xr;
		const float yinteraction = strength * yr;
		const float zinteraction = strength * zr;

		xforce += xinteraction;
		yforce += yinteraction;
		zforce += zinteraction;

		assert(!isnan(xinteraction));
		assert(!isnan(yinteraction));
		assert(!isnan(zinteraction));

		assert(fabs(xinteraction) < 1e4);
		assert(fabs(yinteraction) < 1e4);
		assert(fabs(zinteraction) < 1e4);

		atomicAdd(csolutesacc[soluteid] + sentry    , -xinteraction);
		atomicAdd(csolutesacc[soluteid] + sentry + 1, -yinteraction);
		atomicAdd(csolutesacc[soluteid] + sentry + 2, -zinteraction);
	    }
	}

	write_AOS3f(dst, nunpack, xforce, yforce, zforce);
    }
}

void ComputeContact::halo(ParticlesWrap halos[26], cudaStream_t stream)
{
    NVTX_RANGE("Contact/halo", NVTX_C7);

    int nremote_padded = 0;

    {
	int recvpackcount[26], recvpackstarts_padded[27];

	for(int i = 0; i < 26; ++i)
	    recvpackcount[i] = halos[i].n;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsContact::packcount, recvpackcount,
					   sizeof(recvpackcount), 0, cudaMemcpyHostToDevice, stream));

	recvpackstarts_padded[0] = 0;
	for(int i = 0, s = 0; i < 26; ++i)
	    recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));

	nremote_padded = recvpackstarts_padded[26];

	CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsContact::packstarts_padded, recvpackstarts_padded,
					   sizeof(recvpackstarts_padded), 0, cudaMemcpyHostToDevice, stream));

	const Particle * recvpackstates[26];

	for(int i = 0; i < 26; ++i)
	    recvpackstates[i] = halos[i].p;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsContact::packstates, recvpackstates,
					   sizeof(recvpackstates), 0, cudaMemcpyHostToDevice, stream));

	Acceleration * packresults[26];

	for(int i = 0; i < 26; ++i)
	    packresults[i] = halos[i].a;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsContact::packresults, packresults,
					   sizeof(packresults), 0, cudaMemcpyHostToDevice, stream));
    }

    if(nremote_padded)
    	KernelsContact::halo<<< (nremote_padded + 127) / 128, 128, 0, stream>>>
	    (nremote_padded, cellsentries.size, nsolutes, local_trunk.get_float());

    CUDA_CHECK(cudaPeekAtLastError());
}
