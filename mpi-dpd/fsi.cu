/*
 *  rbc-interactions.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <../dpd-rng.h>

#include "common-kernels.h"
#include "fsi.h"

namespace KernelsFSI
{
    struct Params { float aij, gamma, sigmaf; };

    __constant__ Params params;
}

ComputeFSI::ComputeFSI(MPI_Comm comm)
{
    int myrank;
    MPI_CHECK( MPI_Comm_rank(comm, &myrank));

    local_trunk = Logistic::KISS(1908 - myrank, 1409 + myrank, 290, 12968);

    //TODO: use CUDA_CHECK(cudaEventCreateWithFlags(&evuploaded, cudaEventDisableTiming));

    KernelsFSI::Params params = {0.0f, gammadpd, sigmaf};

    CUDA_CHECK(cudaMemcpyToSymbol(KernelsFSI::params, &params, sizeof(params)));

    CUDA_CHECK(cudaPeekAtLastError());
}

namespace KernelsFSI
{
    texture<float2, cudaTextureType1D> texSolventParticles;
    texture<int, cudaTextureType1D> texCellsStart, texCellsCount;

    bool firsttime = true;

    static const int NCELLS = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;

    __global__  __launch_bounds__(128, 10)
	void interactions_3tpp(const float2 * const particles, const int np, const int nsolvent,
			       float * const acc, float * const accsolvent, const float seed)
    {
#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined! assuming 350
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

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
		count0 = ((cid0 + xcount == NCELLS) ? nsolvent : tex1Dfetch(texCellsStart, cid0 + xcount)) - spidbase;
	    }

	    if (zvalid && ycenter >= 0 && ycenter < YCELLS)
	    {
		const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
		assert(cid1 >= 0 && cid1 + xcount <= NCELLS);
		deltaspid1 = tex1Dfetch(texCellsStart, cid1);
		count1 = ((cid1 + xcount == NCELLS) ? nsolvent : tex1Dfetch(texCellsStart, cid1 + xcount)) - deltaspid1;
	    }

	    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS)
	    {
		const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
		deltaspid2 = tex1Dfetch(texCellsStart, cid2);
		assert(cid2 >= 0 && cid2 + xcount <= NCELLS);
		count2 = ((cid2 + xcount == NCELLS) ? nsolvent : tex1Dfetch(texCellsStart, cid2 + xcount)) - deltaspid2;
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
	    const int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

	    assert(spid >= 0 && spid < nsolvent);

	    const int sentry = 3 * spid;
	    const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry    );
	    const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
	    const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);

	    const float _xr = dst0.x - stmp0.x;
	    const float _yr = dst0.y - stmp0.y;
	    const float _zr = dst1.x - stmp1.x;

	    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

	    const float invrij = rsqrtf(rij2);

	    const float rij = rij2 * invrij;

	    if (rij2 >= 1)
		continue;

	    const float argwr = 1.f - rij;
	    const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

	    const float xr = _xr * invrij;
	    const float yr = _yr * invrij;
	    const float zr = _zr * invrij;

	    const float rdotv =
		xr * (dst1.y - stmp1.y) +
		yr * (dst2.x - stmp2.x) +
		zr * (dst2.y - stmp2.y);

	    const float myrandnr = Logistic::mean0var1(seed, pid, spid);

	    const float strength = params.aij * argwr + (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

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

	    atomicAdd(accsolvent + sentry    , -xinteraction);
	    atomicAdd(accsolvent + sentry + 1, -yinteraction);
	    atomicAdd(accsolvent + sentry + 2, -zinteraction);
	}

	atomicAdd(acc + 3 * pid + 0, xforce);
	atomicAdd(acc + 3 * pid + 1, yforce);
	atomicAdd(acc + 3 * pid + 2, zforce);

	for(int c = 0; c < 3; ++c)
	    assert(!isnan(acc[3 * pid + c]));
    }

    void setup(const Particle * const solvent, const int npsolvent, const int * const cellsstart, const int * const cellscount)
    {
	if (firsttime)
	{
	    texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
	    texCellsStart.filterMode = cudaFilterModePoint;
	    texCellsStart.mipmapFilterMode = cudaFilterModePoint;
	    texCellsStart.normalized = 0;

	    texCellsCount.channelDesc = cudaCreateChannelDesc<int>();
	    texCellsCount.filterMode = cudaFilterModePoint;
	    texCellsCount.mipmapFilterMode = cudaFilterModePoint;
	    texCellsCount.normalized = 0;

	    texSolventParticles.channelDesc = cudaCreateChannelDesc<float2>();
	    texSolventParticles.filterMode = cudaFilterModePoint;
	    texSolventParticles.mipmapFilterMode = cudaFilterModePoint;
	    texSolventParticles.normalized = 0;

	    CUDA_CHECK(cudaFuncSetCacheConfig(interactions_3tpp, cudaFuncCachePreferL1));

	    firsttime = false;
	}

	size_t textureoffset = 0;

	if (npsolvent)
	{
	    CUDA_CHECK(cudaBindTexture(&textureoffset, &texSolventParticles, solvent, &texSolventParticles.channelDesc,
				       sizeof(float) * 6 * npsolvent));
	    assert(textureoffset == 0);
	}

	const int ncells = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;

	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart, &texCellsStart.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);

	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsCount, cellscount, &texCellsCount.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);
    }
}

void ComputeFSI::bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream)
{
    NVTX_RANGE("FSI/bulk", NVTX_C6);

    if (wsolutes.size() == 0)
	return;

    KernelsFSI::setup(wsolvent.p, wsolvent.n, wsolvent.cellsstart, wsolvent.cellscount);

    for(std::vector<ParticlesWrap>::iterator it = wsolutes.begin(); it != wsolutes.end(); ++it)
   	if (it->n)
	    KernelsFSI::interactions_3tpp<<< (3 * it->n + 127) / 128, 128, 0, stream >>>
		((float2 *)it->p, it->n, wsolvent.n, (float *)it->a, (float *)wsolvent.a, local_trunk.get_float());

    CUDA_CHECK(cudaPeekAtLastError());
}

namespace KernelsFSI
{
    __constant__ int packstarts_padded[27], packcount[26];
    __constant__ Particle * packstates[26];
    __constant__ Acceleration * packresults[26];

    __global__ 	void interactions_halo(const int nparticles_padded, const int nsolvent, float * const accsolvent, const float seed)
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

	float xforce = 0, yforce = 0, zforce = 0;

	const int nzplanes = laneid < nunpack ? 3 : 0;

	for(int zplane = 0; zplane < nzplanes; ++zplane)
	{
	    int scan1, scan2, ncandidates, spidbase;
	    int deltaspid1, deltaspid2;

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

		const int NCELLS = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;
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
		    count0 = ((cid0 + xcount == NCELLS) ? nsolvent : tex1Dfetch(texCellsStart, cid0 + xcount)) - spidbase;
		}

		if (zvalid && ycenter >= 0 && ycenter < YCELLS)
		{
		    const int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
		    assert(cid1 >= 0 && cid1 + xcount <= NCELLS);
		    deltaspid1 = tex1Dfetch(texCellsStart, cid1);
		    count1 = ((cid1 + xcount == NCELLS) ? nsolvent : tex1Dfetch(texCellsStart, cid1 + xcount)) - deltaspid1;
		}

		if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS)
		{
		    const int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
		    deltaspid2 = tex1Dfetch(texCellsStart, cid2);
		    assert(cid2 >= 0 && cid2 + xcount <= NCELLS);
		    count2 = ((cid2 + xcount == NCELLS) ? nsolvent : tex1Dfetch(texCellsStart, cid2 + xcount)) - deltaspid2;
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
		const int spid = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

		assert(spid >= 0 && spid < nsolvent);

		const int sentry = 3 * spid;
		const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry    );
		const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
		const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);

		const float _xr = dst0.x - stmp0.x;
		const float _yr = dst0.y - stmp0.y;
		const float _zr = dst1.x - stmp1.x;

		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

		const float invrij = rsqrtf(rij2);

		const float rij = rij2 * invrij;

		if (rij2 >= 1)
		    continue;

		const float argwr = 1.f - rij;
		const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;

		const float rdotv =
		    xr * (dst1.y - stmp1.y) +
		    yr * (dst2.x - stmp2.x) +
		    zr * (dst2.y - stmp2.y);

		const float myrandnr = Logistic::mean0var1(seed, pid, spid);

		const float strength = params.aij * argwr + (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

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

		atomicAdd(accsolvent + sentry    , -xinteraction);
		atomicAdd(accsolvent + sentry + 1, -yinteraction);
		atomicAdd(accsolvent + sentry + 2, -zinteraction);
	    }
	}

	write_AOS3f(dst, nunpack, xforce, yforce, zforce);
    }
}

void ComputeFSI::halo(ParticlesWrap halos[26], cudaStream_t stream)
{
    NVTX_RANGE("FSI/halo", NVTX_C7);

    KernelsFSI::setup(wsolvent.p, wsolvent.n, wsolvent.cellsstart, wsolvent.cellscount);

    int nremote_padded = 0;

    {
	int recvpackcount[26], recvpackstarts_padded[27];

	for(int i = 0; i < 26; ++i)
	    recvpackcount[i] = halos[i].n;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsFSI::packcount, recvpackcount,
					   sizeof(recvpackcount), 0, cudaMemcpyHostToDevice, stream));

	recvpackstarts_padded[0] = 0;
	for(int i = 0, s = 0; i < 26; ++i)
	    recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));

	nremote_padded = recvpackstarts_padded[26];

	CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsFSI::packstarts_padded, recvpackstarts_padded,
					   sizeof(recvpackstarts_padded), 0, cudaMemcpyHostToDevice, stream));
    }

    {
	const Particle * recvpackstates[26];

	for(int i = 0; i < 26; ++i)
	    recvpackstates[i] = halos[i].p;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsFSI::packstates, recvpackstates,
					   sizeof(recvpackstates), 0, cudaMemcpyHostToDevice, stream));
    }

    {
	Acceleration * packresults[26];

	for(int i = 0; i < 26; ++i)
	    packresults[i] = halos[i].a;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsFSI::packresults, packresults,
					   sizeof(packresults), 0, cudaMemcpyHostToDevice, stream));
    }

    if(nremote_padded)
    	KernelsFSI::interactions_halo<<< (nremote_padded + 127) / 128, 128, 0, stream>>>
	    (nremote_padded, wsolvent.n, (float *)wsolvent.a, local_trunk.get_float());

    CUDA_CHECK(cudaPeekAtLastError());
}

