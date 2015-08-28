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

#include <../dpd-rng.h>

#include "fsi.h"

namespace FSI_PUP
{
    __constant__ int sendbagsizes[26], * scattered_indices[26];
}

namespace FSI_CORE
{
    struct Params
    {
	float aij, gamma, sigmaf;
    };

    __constant__ Params params;
}

ComputeFSI::ComputeFSI(MPI_Comm _cartcomm) :
firstpost(true), requiredpacksizes(26), packstarts_padded(27)
{
    assert(XSIZE_SUBDOMAIN % 2 == 0 && YSIZE_SUBDOMAIN % 2 == 0 && ZSIZE_SUBDOMAIN % 2 == 0);
    assert(XSIZE_SUBDOMAIN >= 2 && YSIZE_SUBDOMAIN >= 2 && ZSIZE_SUBDOMAIN >= 2);

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));

    local_trunk = Logistic::KISS(1908 - myrank, 1409 + myrank, 290, 12968);

    const float safety_factor = getenv("HEX_COMM_FACTOR") ? atof(getenv("HEX_COMM_FACTOR")) : 1.2f;

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );

	const int xhalosize = d[0] ? 1 : XSIZE_SUBDOMAIN;
	const int yhalosize = d[1] ? 1 : YSIZE_SUBDOMAIN;
	const int zhalosize = d[2] ? 1 : ZSIZE_SUBDOMAIN;

	const int nhalocells = xhalosize * yhalosize * zhalosize;

	int estimate = numberdensity * safety_factor * nhalocells;
	estimate = 32 * ((estimate + 31) / 32);

	remote[i].expected = estimate;
	remote[i].preserve_resize(estimate);

	local[i].expected = estimate;
	local[i].resize(estimate);

	CUDA_CHECK(cudaMemcpyToSymbol(FSI_PUP::sendbagsizes, &estimate, sizeof(int),
				      sizeof(int) * i, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyToSymbol(FSI_PUP::scattered_indices, &local[i].scattered_indices.data,
				      sizeof(int *), sizeof(int *) * i, cudaMemcpyHostToDevice));
    }

    {
	int s = 0;
	for(int i = 0; i < 26; ++i)
	    s += 32 * ((local[i].scattered_indices.capacity + 31) / 32);

	packbuf.resize(s);
	host_packbuf.resize(s);
    }

    CUDA_CHECK(cudaEventCreateWithFlags(&evPpacked, cudaEventDisableTiming | cudaEventBlockingSync));
    CUDA_CHECK(cudaEventCreateWithFlags(&evAcomputed, cudaEventDisableTiming | cudaEventBlockingSync));

    FSI_CORE::Params params = {12.5 , gammadpd, sigmaf};

    CUDA_CHECK(cudaMemcpyToSymbol(FSI_CORE::params, &params, sizeof(params)));

    CUDA_CHECK(cudaPeekAtLastError());
}

namespace FSI_PUP
{
    __device__ bool failed;

    __device__ int pack_count[26];

    __global__ void init()
    {
	assert(blockDim.x == 26);

	if (threadIdx.x == 0)
	    failed = false;

	pack_count[threadIdx.x] = 0;
    }

    __global__ void scatter_indices(const float2 * const particles, const int nparticles)
    {
	assert(blockDim.x * gridDim.x >= nparticles && blockDim.x == 128);

	const int warpid = threadIdx.x >> 5;
	const int base = 32 * (warpid + 4 * blockIdx.x);
	const int nsrc = min(32, nparticles - base);

	float2 s0, s1, s2;
	read_AOS6f(particles + base, nsrc, s0, s1, s2);

	enum
	{
	    HXSIZE = XSIZE_SUBDOMAIN / 2,
	    HYSIZE = YSIZE_SUBDOMAIN / 2,
	    HZSIZE = ZSIZE_SUBDOMAIN / 2
	};

	const int halocode[3] =
	    {
		-1 + (int)(s0.x >= -HXSIZE + 1) + (int)(s0.x >= HXSIZE - 1),
		-1 + (int)(s0.y >= -HYSIZE + 1) + (int)(s0.y >= HYSIZE - 1),
		-1 + (int)(s1.x >= -HZSIZE + 1) + (int)(s1.x >= HZSIZE - 1)
	    };

	if (halocode[0] == 0 && halocode[1] == 0 && halocode[2] == 0)
	    return;

	const int lane = threadIdx.x & 0x1f;
	const int pid = base + lane;

	if (pid < nparticles)
	{
	    //faces
#pragma unroll 3
	    for(int d = 0; d < 3; ++d)
		if (halocode[d])
		{
		    const int xterm = (halocode[0] * (d == 0) + 2) % 3;
		    const int yterm = (halocode[1] * (d == 1) + 2) % 3;
		    const int zterm = (halocode[2] * (d == 2) + 2) % 3;

		    const int bagid = xterm + 3 * (yterm + 3 * zterm);
		    assert(bagid >= 0 && bagid < 26);

		    const int myid = atomicAdd(pack_count + bagid, 1);

		    if (myid < sendbagsizes[bagid])
			scattered_indices[bagid][myid] = pid;
		}

	    //edges
#pragma unroll 3
	    for(int d = 0; d < 3; ++d)
		if (halocode[(d + 1) % 3] && halocode[(d + 2) % 3])
		{
		    const int xterm = (halocode[0] * (d != 0) + 2) % 3;
		    const int yterm = (halocode[1] * (d != 1) + 2) % 3;
		    const int zterm = (halocode[2] * (d != 2) + 2) % 3;

		    const int bagid = xterm + 3 * (yterm + 3 * zterm);
		    assert(bagid >= 0 && bagid < 26);

		    const int myid = atomicAdd(pack_count + bagid, 1);

		    if (myid < sendbagsizes[bagid])
			scattered_indices[bagid][myid] = pid;
		}

	    //one corner
	    if (halocode[0] && halocode[1] && halocode[2])
	    {
		const int xterm = (halocode[0] + 2) % 3;
		const int yterm = (halocode[1] + 2) % 3;
		const int zterm = (halocode[2] + 2) % 3;

		const int bagid = xterm + 3 * (yterm + 3 * zterm);
		assert(bagid >= 0 && bagid < 26);

		const int myid = atomicAdd(pack_count + bagid, 1);

		if (myid < sendbagsizes[bagid])
		    scattered_indices[bagid][myid] = pid;
	    }
	}
    }

    __device__ int pack_start_padded[27];

    __global__ void tiny_scan(int * const required_packsizes, int * const host_pack_start_padded)
    {
	assert(blockDim.x == 32 && gridDim.x == 1);

	const int tid = threadIdx.x;

	int mycount = 0;

	if (tid < 26)
	{
	    mycount = pack_count[threadIdx.x];
	    
	    required_packsizes[tid] = mycount;
	    
	    if (mycount > sendbagsizes[tid])
		failed = true;
	}
	
	int myscan = mycount = 32 * ((mycount + 31) / 32);
	
	for(int L = 1; L < 32; L <<= 1)
	    myscan += (tid >= L) * __shfl_up(myscan, L) ;
	
	if (tid < 27)
	{
	    pack_start_padded[tid] = myscan - mycount;
	    host_pack_start_padded[tid] = myscan - mycount;
	}
    }

    __constant__ float2 * sendbags[26];

    __global__ void pack(const float2 * const particles, const int nparticles, float2 * const buffer, const float nbuffer)
    {
	assert(blockDim.x == 128);

#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined! assuming 350
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

	if (failed)
	    return;

	const int npack_padded = _ACCESS(pack_start_padded + 26);
	assert(blockDim.x * gridDim.x >= npack_padded);
	       
	const int warpid = threadIdx.x >> 5;
	const int localbase = 32 * (warpid + 4 * blockIdx.x);

	if (localbase >= npack_padded)
	    return;

	const int key9 = 9 * ((int)(localbase >= _ACCESS(pack_start_padded + 9)) +
			      (int)(localbase >= _ACCESS(pack_start_padded + 18)));

	const int key3 = 3 * ((int)(localbase >= _ACCESS(pack_start_padded + key9 + 3)) +
			      (int)(localbase >= _ACCESS(pack_start_padded + key9 + 6)));

	const int key1 =
	    (int)(localbase >= _ACCESS(pack_start_padded + key9 + key3 + 1)) +
	    (int)(localbase >= _ACCESS(pack_start_padded + key9 + key3 + 2));

	const int code = key9 + key3 + key1;

	assert(code >= 0 && code < 26);
	assert(localbase >= pack_start_padded[code] && localbase < pack_start_padded[code + 1]);

	const int packbase = localbase - _ACCESS(pack_start_padded + code);

	assert(packbase >= 0 && packbase < pack_count[code]);

	const int npack = min(32, _ACCESS(pack_count + code) - packbase);

	const int lane = threadIdx.x & 0x1f;

	float2 s0, s1, s2;

	if (lane < npack)
	{
	    const int pid = _ACCESS(scattered_indices[code] + packbase + lane);
	    assert(pid >= 0 && pid < nparticles);

	    const int entry = 3 * pid;

	    s0 = _ACCESS(particles + entry);
	    s1 = _ACCESS(particles + entry + 1);
	    s2 = _ACCESS(particles + entry + 2);

	    s0.x -= ((code + 2) % 3 - 1) * XSIZE_SUBDOMAIN;
	    s0.y -= ((code / 3 + 2) % 3 - 1) * YSIZE_SUBDOMAIN;
	    s1.x -= ((code / 9 + 2) % 3 - 1) * ZSIZE_SUBDOMAIN;
	}

	assert(localbase >= 0 && npack >= 0 && localbase + npack < nbuffer);

	write_AOS6f(buffer + localbase, npack, s0, s1, s2);
    }
}

void ComputeFSI::pack_p(const Particle * const solute, const int nsolute, cudaStream_t stream)
{
    NVTX_RANGE("FSI/pack", NVTX_C4);

    FSI_PUP::init<<< 1, 26, 0, stream >>>();
    FSI_PUP::scatter_indices<<< (nsolute + 127) / 128, 128, 0, stream >>>((float2 *)solute, nsolute);
    FSI_PUP::tiny_scan<<< 1, 32, 0, stream >>>(requiredpacksizes.devptr, packstarts_padded.devptr);
    FSI_PUP::pack<<< (nsolute + 127 + 32 * 26) / 128, 128, 0, stream >>>((float2 *)solute, nsolute, (float2 *)packbuf.data, packbuf.capacity);

    CUDA_CHECK(cudaPeekAtLastError());

    //REMINDER: try out the alternative of downloading the data here after streamsync with event.
    //amount of memory transferred could be unnecessarily large?
}

void ComputeFSI::post_p(const Particle * const solute, const int nsolute, cudaStream_t stream, cudaStream_t downloadstream)
{
    //consolidate the packing
    {
	NVTX_RANGE("FSI/consolidate", NVTX_C5);

	CUDA_CHECK(cudaEventSynchronize(evPpacked));

	for(int i = 0; i < 26; ++i)
	    send_counts[i] = requiredpacksizes.data[i];

	bool packingfailed = false;

	for(int i = 0; i < 26; ++i)
	    if (send_counts[i] > local[i].capacity)
		packingfailed = true;

	if (packingfailed)
	{
	    for(int i = 0; i < 26; ++i)
		local[i].resize(send_counts[i]);

	    int newcapacities[26];
	    for(int i = 0; i < 26; ++i)
		newcapacities[i] = local[i].capacity;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::sendbagsizes, newcapacities, sizeof(newcapacities), 0, cudaMemcpyHostToDevice, stream));

	    int * newindices[26];
	    for(int i = 0; i < 26; ++i)
		newindices[i] = local[i].scattered_indices.data;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::scattered_indices, newindices, sizeof(newindices), 0, cudaMemcpyHostToDevice, stream));

	    packbuf.resize(packstarts_padded.data[26]);
	    host_packbuf.resize(packstarts_padded.data[26]);

	    FSI_PUP::init<<< 1, 26, 0, stream >>>();
	    FSI_PUP::scatter_indices<<< (nsolute + 127) / 128, 128, 0, stream >>>((float2 *)solute, nsolute);
	    FSI_PUP::tiny_scan<<< 1, 32, 0, stream >>>(requiredpacksizes.devptr, packstarts_padded.devptr);
	    FSI_PUP::pack<<< (nsolute + 127 + 32 * 26) / 128, 128, 0, stream >>>((float2 *)solute, nsolute, (float2 *)packbuf.data, packbuf.capacity);

	    CUDA_CHECK(cudaStreamSynchronize(stream));

	    for(int i = 0; i < 26; ++i)
		send_counts[i] = requiredpacksizes.data[i];
	}

	CUDA_CHECK(cudaMemcpyAsync(host_packbuf.data, packbuf.data, sizeof(Particle) * packstarts_padded.data[26], cudaMemcpyDeviceToHost, downloadstream));

	CUDA_CHECK(cudaStreamSynchronize(downloadstream));
    }

    //post the sending of the packs
    {
	NVTX_RANGE("FSI/send", NVTX_C6);

	if (firstpost)
	{
	    _postrecvs();

	    firstpost = false;
	}
	else
	{
	    _wait(reqsendC);
	    _wait(reqsendP);
	    _wait(reqsendA);
	}

	reqsendC.resize(26);

	for(int i = 0; i < 26; ++i)
	    MPI_CHECK( MPI_Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i], TAGBASE_C + i, cartcomm,  i + &reqsendC.front()) );

	for(int i = 0; i < 26; ++i)
	{
	    const int start = packstarts_padded.data[i];
	    const int count = send_counts[i];
	    const int expected = local[i].expected;

	    MPI_Request reqP;
	    MPI_CHECK( MPI_Isend(host_packbuf.data + start, expected * 6, MPI_FLOAT, dstranks[i], TAGBASE_P + i, cartcomm, &reqP) );
	    reqsendP.push_back(reqP);

	    if (count > expected)
	    {
		MPI_Request reqP2;
		MPI_CHECK( MPI_Isend(host_packbuf.data + start + expected, (count - expected) * 6,
				     MPI_FLOAT, dstranks[i], TAGBASE_P2 + i, cartcomm, &reqP2) );

		reqsendP.push_back(reqP2);

		printf("ComputeFSI::post_p ooops rank %d needs to send more than expeted: %d instead of %d\n",
		       myrank, count, expected);
	    }
	}
    }
}

namespace FSI_CORE
{
    texture<float2, cudaTextureType1D> texSolventParticles;
    texture<int, cudaTextureType1D> texCellsStart, texCellsCount;

    static bool firsttime = true;

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

	    const float strength = params.aij * argwr + (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;

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
    }
}

void ComputeFSI::fsi_bulk(const Particle * const solvent, const int nsolvent, Acceleration * accsolvent,
			  const int * const cellsstart_solvent, const int * const cellscount_solvent,
			  const Particle * const solute, const int nsolute, Acceleration * accsolute, cudaStream_t stream)
{
    NVTX_RANGE("FSI/bulk", NVTX_C6);

    FSI_CORE::setup(solvent, nsolvent, cellsstart_solvent, cellscount_solvent);

    if (nsolute)
    {
	const float seed = local_trunk.get_float();

	FSI_CORE::interactions_3tpp<<< (3 * nsolute + 127) / 128, 128, 0, stream >>>
	    ((float2 *)rbcs, nsolute, nsolvent, (float *)accsolute, (float *)accsolvent, seed);
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

namespace FSI_CORE
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

		atomicAdd(accsolvent + sentry    , -xinteraction);
		atomicAdd(accsolvent + sentry + 1, -yinteraction);
		atomicAdd(accsolvent + sentry + 2, -zinteraction);
	    }
	}

	write_AOS3f(dst, nunpack, xforce, yforce, zforce);
    }
}

void ComputeFSI::fsi_halo(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
				      const int * const cellsstart_solvent, const int * const cellscount_solvent,
				      cudaStream_t stream, cudaStream_t uploadstream)
{
    {
	NVTX_RANGE("FSI/recv-p", NVTX_C7);

	_wait(reqrecvC);
	_wait(reqrecvP);

	for(int i = 0; i < 26; ++i)
	{
	    const int count = recv_counts[i];
	    const int expected = remote[i].expected;

	    if (count > expected)
	    {
		remote[i].preserve_resize(count);

		MPI_Status status;
		MPI_CHECK( MPI_Recv(remote[i].hstate.data + expected, (count - expected) * 6, MPI_FLOAT, dstranks[i],
				    TAGBASE_P2 + recv_tags[i], cartcomm, &status) );
	    }
	}
    }

    {
	NVTX_RANGE("FSI/halo", NVTX_C7);

	FSI_CORE::setup(solvent, nparticles, cellsstart_solvent, cellscount_solvent);

	for(int i = 0; i < 26; ++i)
	    CUDA_CHECK(cudaMemcpyAsync(remote[i].dstate.data, remote[i].hstate.data, sizeof(Particle) * remote[i].hstate.size,
					       cudaMemcpyHostToDevice, uploadstream));

	int nremote_padded = 0;

	{
	    static int recvpackcount[26], recvpackstarts_padded[27];

	    for(int i = 0; i < 26; ++i)
		recvpackcount[i] = remote[i].dstate.size;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_CORE::packcount, recvpackcount,
					       sizeof(recvpackcount), 0, cudaMemcpyHostToDevice, uploadstream));

	    recvpackstarts_padded[0] = 0;
	    for(int i = 0, s = 0; i < 26; ++i)
		recvpackstarts_padded[i + 1] = (s += 32 * ((remote[i].dstate.size + 31) / 32));

	    nremote_padded = recvpackstarts_padded[26];

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_CORE::packstarts_padded, recvpackstarts_padded,
					       sizeof(recvpackstarts_padded), 0, cudaMemcpyHostToDevice, uploadstream));
	}

	{
	    static Particle * recvpackstates[26];

	    for(int i = 0; i < 26; ++i)
		recvpackstates[i] = remote[i].dstate.data;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_CORE::packstates, recvpackstates,
					       sizeof(recvpackstates), 0, cudaMemcpyHostToDevice, uploadstream));
	}

	{
	    static Acceleration * packresults[26];

	    for(int i = 0; i < 26; ++i)
		packresults[i] = remote[i].result.devptr;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_CORE::packresults, packresults,
					       sizeof(packresults), 0, cudaMemcpyHostToDevice, uploadstream));
	}

	if(nremote_padded)
	{
	    CUDA_CHECK(cudaStreamSynchronize(uploadstream));

	    FSI_CORE::interactions_halo<<< (nremote_padded + 127) / 128, 128, 0, stream>>>
		(nremote_padded, nparticles, (float *)accsolvent, local_trunk.get_float());
	}

	CUDA_CHECK(cudaEventRecord(evAcomputed, stream));
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeFSI::post_a()
{
    NVTX_RANGE("FSI/send-a", NVTX_C1);

    CUDA_CHECK(cudaEventSynchronize(evAcomputed));

    for(int i = 0, c = 0; i < 26; ++i)
    {
	const int count = recv_counts[i];
	const int expected = remote[i].expected;

	MPI_Request reqA;
	MPI_CHECK( MPI_Isend(remote[i].result.data, expected * 6, MPI_FLOAT, dstranks[i], TAGBASE_A + i, cartcomm, &reqA) );
	reqsendA.push_back(reqA);

	if (count > expected)
	{
	    MPI_Request reqA2;
	    MPI_CHECK( MPI_Isend(remote[i].result.data + expected, (count - expected) * 6,
				 MPI_FLOAT, dstranks[i], TAGBASE_A2 + i, cartcomm, &reqA2) );

	    reqsendA.push_back(reqA2);
	}
    }
}

namespace FSI_PUP
{
    __constant__ float * recvbags[26];

    __global__ void unpack(float * const accelerations, const int nparticles, const int npack_padded)
    {
	assert(blockDim.x * gridDim.x >= 3 * npack_padded && blockDim.x == 128);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	const int pid = gid / 3;

	if (pid >= npack_padded)
	    return;

	const int key9 = 9 * ((int)(pid >= _ACCESS(pack_start_padded + 9)) +
			      (int)(pid >= _ACCESS(pack_start_padded + 18)));

	const int key3 = 3 * ((int)(pid >= _ACCESS(pack_start_padded + key9 + 3)) +
			      (int)(pid >= _ACCESS(pack_start_padded + key9 + 6)));

	const int key1 =
	    (int)(pid >= _ACCESS(pack_start_padded + key9 + key3 + 1)) +
	    (int)(pid >= _ACCESS(pack_start_padded + key9 + key3 + 2));

	const int code = key9 + key3 + key1;

	assert(code >= 0 && code < 26);
	assert(pid >= pack_start_padded[code] && pid < pack_start_padded[code + 1]);

	const int lpid = pid - pack_start_padded[code];

	const int component = gid % 3;

	const float myval = recvbags[code][component +  3 * lpid];
	const int dpid = scattered_indices[code][lpid];

	atomicAdd(accelerations + 3 * dpid + component, myval);
    }
}

void ComputeFSI::merge_a(Acceleration * accsolute, const int nsolute, cudaStream_t stream)
{
    NVTX_RANGE("FSI/merge", NVTX_C2);

    _wait(reqrecvA);

    for(int i = 0; i < 26; ++i)
    {
	const int count = send_counts[i];
	assert(remote[i].capacity >= count);

	const int expected = remote[i].expected;

	if (count > expected)
	{
	    MPI_Status status;
	    MPI_CHECK( MPI_Recv(remote[i].result.data + expected, (count - expected) * 6, MPI_FLOAT, dstranks[i],
				TAGBASE_A2 + recv_tags[i], cartcomm, &status) );
	}
    }

    const int npadded = packstarts_padded.data[26];

    FSI_PUP::unpack<<< (npadded * 3 + 127) / 128, 128, 0, stream >>>((float *)accsolute, nsolute, npadded);

    _postrecvs();
}

ComputeFSI::~ComputeFSI()
{
    MPI_CHECK(MPI_Comm_free(&cartcomm));

    CUDA_CHECK(cudaEventDestroy(evPpacked));
    CUDA_CHECK(cudaEventDestroy(evAcomputed));
}
