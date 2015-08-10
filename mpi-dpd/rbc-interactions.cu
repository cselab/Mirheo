/*
 *  rbc-interactions.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <set>
#include <../dpd-rng.h>

#include "rbc-interactions.h"
#include "minmax-massimo.h"

namespace KernelsRBC
{
    struct ParamsFSI
    {
	float aij, gamma, sigmaf;
    };

    __constant__ ParamsFSI params;

    texture<float2, cudaTextureType1D> texSolventParticles;
    texture<int, cudaTextureType1D> texCellsStart, texCellsCount;

    static bool firsttime = true;

    __global__ void shift_send_particles_kernel(const Particle * const src, const int n, const int code, Particle * const dst)
    {
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	if (gid < n)
	{
	    Particle p = src[gid];

	    for(int c = 0; c < 3; ++c)
		p.x[c] -= d[c] * L[c];

	    dst[gid] = p;
	}
    }

    static const int cmaxnrbcs = 64;
    __constant__ float * csources[cmaxnrbcs], * cdestinations[cmaxnrbcs];
    __constant__ int ccodes[cmaxnrbcs];

    template <bool from_cmem>
    __global__ void shift_all_send_particles(const int nrbcs, const int nvertices,
					     const float ** const dsources, const int * dcodes, float ** const ddestinations)
    {
	const int nfloats_per_rbc = 6 * nvertices;

	assert(nfloats_per_rbc * nrbcs <= blockDim.x * gridDim.x);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= nfloats_per_rbc * nrbcs)
	    return;

	const int idrbc = gid / nfloats_per_rbc;
	assert(idrbc < nrbcs);

	const int offset = gid % nfloats_per_rbc;

	float val;
	if (from_cmem)
	    val = csources[idrbc][offset];
	else
	    val = dsources[idrbc][offset];

	int code;
	if (from_cmem)
	    code = ccodes[idrbc];
	else
	    code = dcodes[idrbc];

	const int c = gid % 6;

	val -=
	    (c == 0) * ((code     + 2) % 3 - 1) * XSIZE_SUBDOMAIN +
	    (c == 1) * ((code / 3 + 2) % 3 - 1) * YSIZE_SUBDOMAIN +
	    (c == 2) * ((code / 9 + 2) % 3 - 1) * ZSIZE_SUBDOMAIN ;

	if (from_cmem)
	    cdestinations[idrbc][offset] = val;
	else
	    ddestinations[idrbc][offset] = val;
    }

    SimpleDeviceBuffer<float *> _ddestinations;
    SimpleDeviceBuffer<const float *> _dsources;
    SimpleDeviceBuffer<int> _dcodes;

    void dispose()
    {
	_ddestinations.dispose();
	_dsources.dispose();
	_dcodes.dispose();
    }

    void shift_send_particles(cudaStream_t stream, const int nrbcs, const int nvertices,
			      const float ** const sources, const int * codes, float ** const destinations)
    {
	if (nrbcs == 0)
	    return;

	const int nthreads = nrbcs * nvertices * 6;

	if (nrbcs < cmaxnrbcs)
	{
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(ccodes, codes, sizeof(int) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(cdestinations, destinations, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(csources, sources, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));

	    shift_all_send_particles<true><<<(nthreads + 127) / 128, 128, 0, stream>>>
		(nrbcs, nvertices, NULL, NULL, NULL);

	    CUDA_CHECK(cudaPeekAtLastError());
	}
	else
	{
	    _dcodes.resize(nrbcs);
	    _ddestinations.resize(nrbcs);
	    _dsources.resize(nrbcs);

	    CUDA_CHECK(cudaMemcpyAsync(_dcodes.data, codes, sizeof(int) * nrbcs, cudaMemcpyHostToDevice, stream));
	    CUDA_CHECK(cudaMemcpyAsync(_ddestinations.data, destinations, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));
	    CUDA_CHECK(cudaMemcpyAsync(_dsources.data, sources, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));

	    shift_all_send_particles<false><<<(nthreads + 127) / 128, 128, 0, stream>>>
		(nrbcs, nvertices, _dsources.data, _dcodes.data, _ddestinations.data);
	}
    }

    template <bool from_cmem>
    __global__ void merge_all_acc(const int nrbcs, const int nvertices,
				  const float ** const dsources, float ** const ddestinations)
    {
	if (nrbcs == 0)
	    return;

	const int nfloats_per_rbc = 3 * nvertices;

	assert(nfloats_per_rbc * nrbcs <= blockDim.x * gridDim.x);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= nfloats_per_rbc * nrbcs)
	    return;

	const int idrbc = gid / nfloats_per_rbc;
	assert(idrbc < nrbcs);

	const int offset = gid % nfloats_per_rbc;

	float val;
	if (from_cmem)
	    val = csources[idrbc][offset];
	else
	    val = dsources[idrbc][offset];

	if (from_cmem)
	    atomicAdd(cdestinations[idrbc] + offset, val);
	else
	    atomicAdd(ddestinations[idrbc] + offset, val);
    }

    void merge_all_accel(cudaStream_t stream, const int nrbcs, const int nvertices,
			 const float ** const sources, float ** const destinations)
    {
	if (nrbcs == 0)
	    return;

	const int nthreads = nrbcs * nvertices * 3;

	CUDA_CHECK(cudaPeekAtLastError());

	if (nrbcs < cmaxnrbcs)
	{
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(cdestinations, destinations, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(csources, sources, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));

	    merge_all_acc<true><<<(nthreads + 127) / 128, 128, 0, stream>>>(nrbcs, nvertices, NULL, NULL);

	    CUDA_CHECK(cudaPeekAtLastError());
	}
	else
	{
	    _ddestinations.resize(nrbcs);
	    _dsources.resize(nrbcs);

	    CUDA_CHECK(cudaMemcpyAsync(_ddestinations.data, destinations, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));
	    CUDA_CHECK(cudaMemcpyAsync(_dsources.data, sources, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));

	    merge_all_acc<false><<<(nthreads + 127) / 128, 128, 0, stream>>>(nrbcs, nvertices, _dsources.data, _ddestinations.data);
	}
    }

#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined! assuming 350
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

    static const int NCELLS = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;

    __global__  __launch_bounds__(128, 10)
	void interactions_3tpp(const float2 * const particles, const int np, const int nsolvent,
			       float * const acc, float * const accsolvent, const float seed)
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

	    const float strength = aij * argwr + (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

	    const float xinteraction = strength * xr;
	    const float yinteraction = strength * yr;
	    const float zinteraction = strength * zr;

	    xforce += xinteraction;
	    yforce += yinteraction;
	    zforce += zinteraction;

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

    __constant__ int packstarts_padded[27], packcount[26];
    __constant__ Particle * packstates[26];
    __constant__ Acceleration * packresults[26];

    __global__ 
	void interactions_halo(const int nparticles_padded,
			       const int nsolvent, float * const accsolvent, const float seed)
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
	    assert(code >= 1 && code < 27);
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

		const float strength = aij * argwr + (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

		const float xinteraction = strength * xr;
		const float yinteraction = strength * yr;
		const float zinteraction = strength * zr;

		xforce += xinteraction;
		yforce += yinteraction;
		zforce += zinteraction;

		atomicAdd(accsolvent + sentry    , -xinteraction);
		atomicAdd(accsolvent + sentry + 1, -yinteraction);
		atomicAdd(accsolvent + sentry + 2, -zinteraction);
	    }
	}

	write_AOS3f(dst, nunpack, xforce, yforce, zforce);
    }

    void setup(const Particle * const solvent, const int npsolvent, const int * const cellsstart, const int * const cellscount,
	       const Particle * const solute, const int npsolute, const int * const solute_cellsstart, const int * const solute_cellscount)
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

	    firsttime = false;
	}

	size_t textureoffset;
	if (npsolvent)
	    CUDA_CHECK(cudaBindTexture(&textureoffset, &texSolventParticles, solvent, &texSolventParticles.channelDesc,
				       sizeof(float) * 6 * npsolvent));
	assert(textureoffset == 0);

	const int ncells = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;

	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart, &texCellsStart.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsCount, cellscount, &texCellsCount.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);

	CUDA_CHECK(cudaFuncSetCacheConfig(interactions_3tpp, cudaFuncCachePreferL1));
	//CUDA_CHECK(cudaFuncSetCacheConfig(fsi_forces_old, cudaFuncCachePreferL1));
    }
}

ComputeInteractionsRBC::ComputeInteractionsRBC(MPI_Comm _cartcomm):
nvertices(0)
{
    assert(XSIZE_SUBDOMAIN % 2 == 0 && YSIZE_SUBDOMAIN % 2 == 0 && ZSIZE_SUBDOMAIN % 2 == 0);
    assert(XSIZE_SUBDOMAIN >= 2 && YSIZE_SUBDOMAIN >= 2 && ZSIZE_SUBDOMAIN >= 2);

    if (rbcs)
    {
	CudaRBC::Extent host_extent;
	CudaRBC::setup(nvertices, host_extent);
    }

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));

    local_trunk = Logistic::KISS(1908 - myrank, 1409 + myrank, 290, 12968);

    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));

    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );
    }

    KernelsRBC::ParamsFSI params = {12.5 , gammadpd, sigmaf};

    CUDA_CHECK(cudaMemcpyToSymbol(KernelsRBC::params, &params, sizeof(KernelsRBC::ParamsFSI)));

    CUDA_CHECK(cudaEventCreate(&evextents, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&evfsi, cudaEventDisableTiming));
}

void ComputeInteractionsRBC::_compute_extents(const Particle * const rbcs, const int nrbcs, cudaStream_t stream)
{
#if 1
    if (nrbcs)
	minmax_massimo(rbcs, nvertices, nrbcs, minextents.devptr, maxextents.devptr, stream);
#else
    for(int i = 0; i < nrbcs; ++i)
	CudaRBC::extent_nohost(stream, (float *)(rbcs + nvertices * i), extents.devptr + i);
#endif
}

void ComputeInteractionsRBC::extent(const Particle * const rbcs, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RBC/extent", NVTX_C2);

    minextents.resize(nrbcs);
    maxextents.resize(nrbcs);

    _compute_extents(rbcs, nrbcs, stream);

    CUDA_CHECK(cudaEventRecord(evextents, stream));
}

void ComputeInteractionsRBC::count(const int nrbcs)
{
    NVTX_RANGE("RBC/count", NVTX_C3);

    CUDA_CHECK(cudaEventSynchronize(evextents));

    for(int i = 0; i < 26; ++i)
	haloreplica[i].clear();

    for(int i = 0; i < nrbcs; ++i)
    {
	float pmin[3] = { minextents.data[i].x, minextents.data[i].y, minextents.data[i].z };
	float pmax[3] = { maxextents.data[i].x, maxextents.data[i].y, maxextents.data[i].z };

	for(int code = 0; code < 26; ++code)
	{
	    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	    const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	    bool interacting = true;

	    for(int c = 0; c < 3; ++c)
	    {
		const float range_start = max((float)(d[c] * L[c] - L[c]/2 - 1), pmin[c]);
		const float range_end = min((float)(d[c] * L[c] + L[c]/2 + 1), pmax[c]);

		interacting &= range_end > range_start;
	    }

	    if (interacting)
		haloreplica[code].push_back(i);
	}
    }

    for(int i = 0; i <26; ++i)
	MPI_CHECK(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i], recv_tags[i] + 2077, cartcomm, reqrecvcounts + i));


    for(int i = 0; i < 26; ++i)
    {
	send_counts[i] = haloreplica[i].size();
	MPI_CHECK(MPI_Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i], i + 2077, cartcomm, reqsendcounts + i));
    }

    for(int i = 0; i < 26; ++i)
	local[i].setup(send_counts[i] * nvertices);
}

void ComputeInteractionsRBC::exchange_count()
{
    NVTX_RANGE("RBC/exchange-count", NVTX_C4);

    MPI_Status statuses[26];
    MPI_CHECK(MPI_Waitall(26, reqrecvcounts, statuses));
    MPI_CHECK(MPI_Waitall(26, reqsendcounts, statuses));

    for(int i = 0; i < 26; ++i)
	remote[i].setup(recv_counts[i] * nvertices);
}

void ComputeInteractionsRBC::pack_p(const Particle * const rbcs, cudaStream_t stream)
{
    NVTX_RANGE("RBC/pack", NVTX_C4);

#if 1
    {
	static std::vector<int> codes;
	static std::vector<const float *> src;
	static std::vector<float *> dst;

	codes.clear();
	src.clear();
	dst.clear();

	for(int i = 0; i < 26; ++i)
	    for(int j = 0; j < haloreplica[i].size(); ++j)
	    {
		codes.push_back(i);
		src.push_back((float *)(rbcs + nvertices * haloreplica[i][j]));
		dst.push_back((float *)(local[i].state.devptr + nvertices * j));
	    }

	KernelsRBC::shift_send_particles(stream, src.size(), nvertices, &src.front(), &codes.front(), &dst.front());

	CUDA_CHECK(cudaPeekAtLastError());
    }
#else
    for(int i = 0; i < 26; ++i)
    {
	for(int j = 0; j < haloreplica[i].size(); ++j)
	    KernelsRBC::shift_send_particles<<< (nvertices + 127) / 128, 128, 0, stream>>>
		(rbcs + nvertices * haloreplica[i][j], nvertices, i, local[i].state.devptr + nvertices * j);

	CUDA_CHECK(cudaPeekAtLastError());
    }
#endif

    CUDA_CHECK(cudaEventRecord(evfsi, stream));
}

void ComputeInteractionsRBC::post_p()
{
    NVTX_RANGE("RBC/post-p", NVTX_C5);

    CUDA_CHECK(cudaEventSynchronize(evfsi));

    for(int i = 0; i < 26; ++i)
	if (recv_counts[i] > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(remote[i].state.data, recv_counts[i] * nvertices, Particle::datatype(), dstranks[i],
				recv_tags[i] + 2011, cartcomm, &request));

	    reqrecvp.push_back(request);
	}

    for(int i = 0; i < 26; ++i)
	if (send_counts[i] > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(local[i].result.data, send_counts[i] * nvertices, Acceleration::datatype(), dstranks[i],
				recv_tags[i] + 2285, cartcomm, &request));

	    reqrecvacc.push_back(request);

	    MPI_CHECK(MPI_Isend(local[i].state.data, send_counts[i] * nvertices, Particle::datatype(), dstranks[i],
				i + 2011, cartcomm, &request));

	    reqsendp.push_back(request);
	}
}

void ComputeInteractionsRBC::internal_forces(const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
{
    CudaRBC::forces_nohost(stream, nrbcs, (float *)rbcs, (float *)accrbc);
}

void ComputeInteractionsRBC::fsi_bulk(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
				      const int * const cellsstart_solvent, const int * const cellscount_solvent,
				      const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
{
    NVTX_RANGE("RBC/fsi-bulk", NVTX_C6);

    const int nsolute = nrbcs * nvertices;
    const int nsolvent = nparticles;

    KernelsRBC::setup(solvent, nparticles, cellsstart_solvent, cellscount_solvent,
		      NULL, 0, NULL, NULL);

    if (nrbcs > 0 && nparticles > 0)
    {
	const float seed = local_trunk.get_float();

	KernelsRBC::interactions_3tpp<<< (3 * nsolute + 127) / 128, 128, 0, stream >>>
	    ((float2 *)rbcs, nsolute, nsolvent, (float *)accrbc, (float *)accsolvent, seed);
    }
}

void ComputeInteractionsRBC::fsi_halo(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
				      const int * const cellsstart_solvent, const int * const cellscount_solvent,
				      const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
{
    NVTX_RANGE("RBC/fsi-halo", NVTX_C7);

    KernelsRBC::setup(solvent, nparticles, cellsstart_solvent, cellscount_solvent,
		      NULL, 0, NULL, NULL);
    _wait(reqrecvp);
    _wait(reqsendp);

#if 1
    {
	int nremote_padded = 0;

	{
	    static int packcount[26], packstarts_padded[27];

	    for(int i = 0; i < 26; ++i)
		packcount[i] = remote[i].state.size;
	    
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsRBC::packcount, packcount,
					       sizeof(packcount), 0, cudaMemcpyHostToDevice, stream));
	    
	    packstarts_padded[0] = 0;
	    for(int i = 0, s = 0; i < 26; ++i)
		packstarts_padded[i + 1] = (s += 32 * ((remote[i].state.size + 31) / 32));

	    nremote_padded = packstarts_padded[26];
	    
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsRBC::packstarts_padded, packstarts_padded,
					       sizeof(packstarts_padded), 0, cudaMemcpyHostToDevice, stream));
	}

	{
	    static Particle * packstates[26];

	    for(int i = 0; i < 26; ++i)
		packstates[i] = remote[i].state.devptr;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsRBC::packstates, packstates,
					       sizeof(packstates), 0, cudaMemcpyHostToDevice, stream));
	}

	{
	    static Acceleration * packresults[26];

	    for(int i = 0; i < 26; ++i)
		packresults[i] = remote[i].result.devptr;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(KernelsRBC::packresults, packresults,
					       sizeof(packresults), 0, cudaMemcpyHostToDevice, stream));
	}

	if(nremote_padded)
	    KernelsRBC::interactions_halo<<< (nremote_padded + 127) / 128, 128, 0, stream>>>
		(nremote_padded, nparticles, (float *)accsolvent, local_trunk.get_float());
    }
#else
    for(int i = 0; i < 26; ++i)
    {
	const int count = remote[i].state.size;

	if (count > 0)
	    KernelsRBC::fsi_forces<<< (count + 127) / 128, 128, 0, stream >>>
		(local_trunk.get_float(), accsolvent, nparticles, remote[i].state.devptr, count, remote[i].result.devptr);
    }
#endif

    CUDA_CHECK(cudaEventRecord(evfsi));

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeInteractionsRBC::post_a()
{
    NVTX_RANGE("RBC/send-results", NVTX_C1);

    CUDA_CHECK(cudaEventSynchronize(evfsi));

    _wait(reqsendacc);

    for(int i = 0; i < 26; ++i)
	if (recv_counts[i] > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Isend(remote[i].result.data, recv_counts[i] * nvertices, Acceleration::datatype(), dstranks[i],
				i + 2285, cartcomm, &request));

	    reqsendacc.push_back(request);
	}
}

void ComputeInteractionsRBC::merge_a(Acceleration * accrbc, cudaStream_t stream)
{
    NVTX_RANGE("RBC/merge", NVTX_C2);

    _wait(reqrecvacc);

#if 1
    {
	static std::vector<const float *> src;
	static std::vector<float *> dst;

	src.clear();
	dst.clear();

	for(int i = 0; i < 26; ++i)
	    for(int j = 0; j < haloreplica[i].size(); ++j)
	    {
		src.push_back((float *)(local[i].result.devptr + nvertices * j));
		dst.push_back((float *)(accrbc + nvertices * haloreplica[i][j]));
	    }

	KernelsRBC::merge_all_accel(stream, src.size(), nvertices, &src.front(), &dst.front());

	CUDA_CHECK(cudaPeekAtLastError());
    }
#else
    for(int i = 0; i < 26; ++i)
	for(int j = 0; j < haloreplica[i].size(); ++j)
	    KernelsRBC::merge_accelerations<<< (nvertices + 127) / 128, 128, 0, stream>>>(local[i].result.devptr + nvertices * j, nvertices,
											  accrbc + nvertices * haloreplica[i][j]);
#endif
}

ComputeInteractionsRBC::~ComputeInteractionsRBC()
{
    MPI_CHECK(MPI_Comm_free(&cartcomm));

    CUDA_CHECK(cudaEventDestroy(evextents));
    CUDA_CHECK(cudaEventDestroy(evfsi));

    KernelsRBC::dispose();
}
