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
    __constant__ int ccapacities[26], * scattered_indices[26];
}

namespace FSI_CORE
{
    struct Params { float aij, gamma, sigmaf; };

    __constant__ Params params;
}

ComputeFSI::ComputeFSI(MPI_Comm _cartcomm) :
iterationcount(-1), packstotalstart(27), host_packstotalstart(27), host_packstotalcount(26)
{
    assert(XSIZE_SUBDOMAIN % 2 == 0 && YSIZE_SUBDOMAIN % 2 == 0 && ZSIZE_SUBDOMAIN % 2 == 0);
    assert(XSIZE_SUBDOMAIN >= 4 && YSIZE_SUBDOMAIN >= 4 && ZSIZE_SUBDOMAIN >= 4);

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));

    local_trunk = Logistic::KISS(1908 - myrank, 1409 + myrank, 290, 12968);

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );

	const int estimate = 1;
	remote[i].preserve_resize(estimate);
	local[i].resize(estimate);
	local[i].update();

	CUDA_CHECK(cudaMemcpyToSymbol(FSI_PUP::ccapacities, &estimate, sizeof(int),
				      sizeof(int) * i, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyToSymbol(FSI_PUP::scattered_indices, &local[i].scattered_indices.data,
				      sizeof(int *), sizeof(int *) * i, cudaMemcpyHostToDevice));
    }

    _adjust_packbuffers();

    CUDA_CHECK(cudaEventCreateWithFlags(&evPpacked, cudaEventDisableTiming | cudaEventBlockingSync));
    CUDA_CHECK(cudaEventCreateWithFlags(&evAcomputed, cudaEventDisableTiming | cudaEventBlockingSync));

    FSI_CORE::Params params = {12.5 , gammadpd, sigmaf};

    CUDA_CHECK(cudaMemcpyToSymbol(FSI_CORE::params, &params, sizeof(params)));

    CUDA_CHECK(cudaPeekAtLastError());
}

namespace FSI_PUP
{
    __device__ bool failed;

    __global__ void init() { failed = false; }

    __constant__ int coffsets[26];

    __global__ void scatter_indices(const float2 * const particles, const int nparticles, int * const counts)
    {
	assert(blockDim.x * gridDim.x >= nparticles && blockDim.x == 128);

	const int warpid = threadIdx.x >> 5;
	const int base = 32 * (warpid + 4 * blockIdx.x);
	const int nsrc = min(32, nparticles - base);

	float2 s0, s1, s2;
	read_AOS6f(particles + 3 * base, nsrc, s0, s1, s2);

	const int lane = threadIdx.x & 0x1f;
	const int pid = base + lane;

	if (pid >= nparticles)
	    return;

	enum
	{
	    HXSIZE = XSIZE_SUBDOMAIN / 2,
	    HYSIZE = YSIZE_SUBDOMAIN / 2,
	    HZSIZE = ZSIZE_SUBDOMAIN / 2
	};

	assert(s0.x >= -HXSIZE + 1 - XSIZE_SUBDOMAIN && s0.x < HXSIZE - 1 + XSIZE_SUBDOMAIN);
	assert(s0.y >= -HYSIZE + 1 - YSIZE_SUBDOMAIN && s0.y < HYSIZE - 1 + YSIZE_SUBDOMAIN);
	assert(s1.x >= -HZSIZE + 1 - ZSIZE_SUBDOMAIN && s1.x < HZSIZE - 1 + ZSIZE_SUBDOMAIN);

	const int halocode[3] =
	    {
		-1 + (int)(s0.x >= -HXSIZE + 1) + (int)(s0.x >= HXSIZE - 1),
		-1 + (int)(s0.y >= -HYSIZE + 1) + (int)(s0.y >= HYSIZE - 1),
		-1 + (int)(s1.x >= -HZSIZE + 1) + (int)(s1.x >= HZSIZE - 1)
	    };

	if (halocode[0] == 0 && halocode[1] == 0 && halocode[2] == 0)
	    return;

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

		const int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

		if (myid < ccapacities[bagid])
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

		const int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

		if (myid < ccapacities[bagid])
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

	    const int myid = coffsets[bagid] + atomicAdd(counts + bagid, 1);

	    if (myid < ccapacities[bagid])
		scattered_indices[bagid][myid] = pid;
	}
    }

    __global__ void tiny_scan(const int * const counts, const int * const oldtotalcounts, int * const totalcounts, int * const paddedstarts)
    {
	assert(blockDim.x == 32 && gridDim.x == 1);

	const int tid = threadIdx.x;

	int mycount = 0;

	if (tid < 26)
	{
	    mycount = counts[tid];

	    if (mycount > ccapacities[tid])
		failed = true;

	    if (totalcounts && oldtotalcounts)
	    {
		const int newcount = mycount + oldtotalcounts[tid];
		totalcounts[tid] = newcount;
	    }
	}

	if (paddedstarts)
	{
	    int myscan = mycount = 32 * ((mycount + 31) / 32);

	    for(int L = 1; L < 32; L <<= 1)
		myscan += (tid >= L) * __shfl_up(myscan, L) ;

	    if (tid < 27)
		paddedstarts[tid] = myscan - mycount;
	}
    }

    __constant__ int ccounts[26], cbases[27], cpaddedstarts[27];

    __global__ void pack(const float2 * const particles, const int nparticles, float2 * const buffer, const int nbuffer, const int soluteid)
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

	const int npack_padded = cpaddedstarts[26];
	assert(blockDim.x * gridDim.x >= npack_padded);

	const int warpid = threadIdx.x >> 5;
	const int localbase = 32 * (warpid + 4 * blockIdx.x);

	if (localbase >= npack_padded)
	    return;

	const int key9 = 9 * ((int)(localbase >= cpaddedstarts[9]) +
			      (int)(localbase >= cpaddedstarts[18]));

	const int key3 = 3 * ((int)(localbase >= cpaddedstarts[key9 + 3]) +
			      (int)(localbase >= cpaddedstarts[key9 + 6]));

	const int key1 =
	    (int)(localbase >= cpaddedstarts[key9 + key3 + 1]) +
	    (int)(localbase >= cpaddedstarts[key9 + key3 + 2]);

	const int code = key9 + key3 + key1;

	assert(code >= 0 && code < 26);
	assert(localbase >= cpaddedstarts[code] && localbase < cpaddedstarts[code + 1]);

	const int packbase = localbase - cpaddedstarts[code];
	assert(packbase >= 0 && packbase < ccounts[code]);

	const int npack = min(32, ccounts[code] - packbase);

	const int lane = threadIdx.x & 0x1f;

	float2 s0, s1, s2;

	if (lane < npack)
	{
	    const int pid = _ACCESS(scattered_indices[code] + coffsets[code] + packbase + lane);

	    if (pid < 0)
		printf("ooops solutid %d: coffsets[%d]=%d, packbase= %d, lane=%d\n", soluteid, code, coffsets[code], packbase, lane);

	    assert(pid >= 0);
	    assert(pid < nparticles);

	    const int entry = 3 * pid;

	    s0 = _ACCESS(particles + entry);
	    s1 = _ACCESS(particles + entry + 1);
	    s2 = _ACCESS(particles + entry + 2);

	    s0.x -= ((code + 2) % 3 - 1) * XSIZE_SUBDOMAIN;
	    s0.y -= ((code / 3 + 2) % 3 - 1) * YSIZE_SUBDOMAIN;
	    s1.x -= ((code / 9 + 2) % 3 - 1) * ZSIZE_SUBDOMAIN;
	}

	assert(cbases[code] + coffsets[code] + packbase >= 0 && npack >= 0);
	assert(cbases[code] + coffsets[code] + packbase + npack < nbuffer);

	write_AOS6f(buffer + 3 * (cbases[code] + coffsets[code] + packbase), npack, s0, s1, s2);
    }
}

void ComputeFSI::_pack_attempt(cudaStream_t stream)
{
    CUDA_CHECK(cudaMemsetAsync(packscount.data, 0, sizeof(int) * packscount.size, stream));
    CUDA_CHECK(cudaMemsetAsync(packsoffset.data, 0, sizeof(int) * packsoffset.size, stream));
    CUDA_CHECK(cudaMemsetAsync(packsstart.data, 0, sizeof(int) * packsstart.size, stream));

    FSI_PUP::init<<< 1, 1, 0, stream >>>();

    for(int i = 0; i < wsolutes.size(); ++i)
    {
	const ParticlesWrap it = wsolutes[i];

       	if (it.n)
	{
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::coffsets, packsoffset.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));

	    FSI_PUP::scatter_indices<<< (it.n + 127) / 128, 128, 0, stream >>>((float2 *)it.p, it.n, packscount.data + i * 26);
	}

	FSI_PUP::tiny_scan<<< 1, 32, 0, stream >>>(packscount.data + i * 26, packsoffset.data + 26 * i,
						   packsoffset.data + 26 * (i + 1), packsstart.data + i * 27);

	CUDA_CHECK(cudaPeekAtLastError());
    }

    CUDA_CHECK(cudaMemcpyAsync(host_packstotalcount.data, packsoffset.data + 26 * wsolutes.size(), sizeof(int) * 26, cudaMemcpyDeviceToHost, stream));

    FSI_PUP::tiny_scan<<< 1, 32, 0, stream >>>(packsoffset.data + 26 * wsolutes.size(), NULL, NULL, packstotalstart.data);

    CUDA_CHECK(cudaMemcpyAsync(host_packstotalstart.data, packstotalstart.data, sizeof(int) * 27, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::cbases, packstotalstart.data, sizeof(int) * 27, 0, cudaMemcpyDeviceToDevice, stream));

    for(int i = 0; i < wsolutes.size(); ++i)
    {
	const ParticlesWrap it = wsolutes[i];

	if (it.n)
	{
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::coffsets, packsoffset.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::ccounts, packscount.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::cpaddedstarts, packsstart.data + 27 * i, sizeof(int) * 27, 0, cudaMemcpyDeviceToDevice, stream));

	    FSI_PUP::pack<<< (it.n + 127 + 31 * 26) / 128, 128, 0, stream >>>((float2 *)it.p, it.n, (float2 *)packbuf.data, packbuf.capacity, i);
	}
    }

    CUDA_CHECK(cudaEventRecord(evPpacked, stream));

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeFSI::pack_p(cudaStream_t stream)
{
    if (wsolutes.size() == 0)
	return;

    NVTX_RANGE("FSI/pack", NVTX_C4);

    ++iterationcount;

    packscount.resize(26 * wsolutes.size());
    packsoffset.resize(26 * (wsolutes.size() + 1));
    packsstart.resize(27 * wsolutes.size());

    _pack_attempt(stream);
}

void ComputeFSI::post_p(cudaStream_t stream, cudaStream_t downloadstream)
{
    if (wsolutes.size() == 0)
	return;

    //consolidate the packing
    {
	NVTX_RANGE("FSI/consolidate", NVTX_C5);

	CUDA_CHECK(cudaEventSynchronize(evPpacked));

	if (iterationcount == 0)
	    _postrecvC();
	else
	    _wait(reqsendC);

	for(int i = 0; i < 26; ++i)
	    send_counts[i] = host_packstotalcount.data[i];

	bool packingfailed = false;

	for(int i = 0; i < 26; ++i)
	    packingfailed |= send_counts[i] > local[i].capacity();

	if (packingfailed)
	{
	    for(int i = 0; i < 26; ++i)
		local[i].resize(send_counts[i]);

	    int newcapacities[26];
	    for(int i = 0; i < 26; ++i)
		newcapacities[i] = local[i].capacity();

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::ccapacities, newcapacities, sizeof(newcapacities), 0, cudaMemcpyHostToDevice, stream));

	    int * newindices[26];
	    for(int i = 0; i < 26; ++i)
		newindices[i] = local[i].scattered_indices.data;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::scattered_indices, newindices, sizeof(newindices), 0, cudaMemcpyHostToDevice, stream));

	    _adjust_packbuffers();

	    _pack_attempt(stream);

	    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifndef NDEBUG
	    for(int i = 0; i < 26; ++i)
		assert(send_counts[i] == host_packstotalcount.data[i]);

	    for(int i = 0; i < 26; ++i)
		assert(send_counts[i] <= local[i].capacity());
#endif
	}

	//for(int i = 0; i < 26; ++i)
	//  printf("packing %d : %d\n", i, send_counts[i]);

	for(int i = 0; i < 26; ++i)
	    local[i].resize(send_counts[i]);

	_postrecvA();

	if (iterationcount == 0)
	    _postrecvP();
	else
	    _wait(reqsendP);

	if (host_packstotalstart.data[26])
	{
	    assert(host_packbuf.capacity >= packbuf.capacity);
	    CUDA_CHECK(cudaMemcpyAsync(host_packbuf.data, packbuf.data, sizeof(Particle) * host_packstotalstart.data[26],
				       cudaMemcpyDeviceToHost, downloadstream));
	}

	CUDA_CHECK(cudaStreamSynchronize(downloadstream));
    }

    //post the sending of the packs
    {
	NVTX_RANGE("FSI/send", NVTX_C6);

	reqsendC.resize(26);

	for(int i = 0; i < 26; ++i)
	    MPI_CHECK( MPI_Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i], TAGBASE_C + i, cartcomm,  &reqsendC[i]) );

	for(int i = 0; i < 26; ++i)
	{
	    const int start = host_packstotalstart.data[i];
	    const int count = send_counts[i];
	    const int expected = local[i].expected();

	    MPI_Request reqP;
	    MPI_CHECK( MPI_Isend(host_packbuf.data + start, expected * 6, MPI_FLOAT, dstranks[i], TAGBASE_P + i, cartcomm, &reqP) );
	    reqsendP.push_back(reqP);

	    if (count > expected)
	    {
		MPI_Request reqP2;
		MPI_CHECK( MPI_Isend(host_packbuf.data + start + expected, (count - expected) * 6,
				     MPI_FLOAT, dstranks[i], TAGBASE_P2 + i, cartcomm, &reqP2) );

		reqsendP.push_back(reqP2);
#if 0
		printf("ComputeFSI::post_p ooops rank %d needs to send more than expected: %d instead of %d (i %d)\n",
		       myrank, count, expected, i);
		fflush(stdout);
#endif
	    }
	}
    }
}

namespace FSI_CORE
{
    texture<float2, cudaTextureType1D> texSolventParticles;
    texture<int, cudaTextureType1D> texCellsStart, texCellsCount;

    bool firsttime = true;

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

void ComputeFSI::bulk(cudaStream_t stream)
{
    if (wsolutes.size() == 0)
	return;

    NVTX_RANGE("FSI/bulk", NVTX_C6);

    FSI_CORE::setup(wsolvent.p, wsolvent.n, wsolvent.cellsstart, wsolvent.cellscount);

    for(std::vector<ParticlesWrap>::iterator it = wsolutes.begin(); it != wsolutes.end(); ++it)
    {
	const float seed = local_trunk.get_float();

	if (it->n)
	    FSI_CORE::interactions_3tpp<<< (3 * it->n + 127) / 128, 128, 0, stream >>>
		((float2 *)it->p, it->n, wsolvent.n, (float *)it->a, (float *)wsolvent.a, seed);
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

void ComputeFSI::halo(cudaStream_t stream, cudaStream_t uploadstream)
{
    if (wsolutes.size() == 0)
	return;

    {
	NVTX_RANGE("FSI/recv-p", NVTX_C7);

	_wait(reqrecvC);
	_wait(reqrecvP);

	for(int i = 0; i < 26; ++i)
	{
	    const int count = recv_counts[i];
	    const int expected = remote[i].expected();

	    remote[i].preserve_resize(count);

	    MPI_Status status;

	    if (count > expected)
		MPI_CHECK( MPI_Recv(remote[i].hstate.data + expected, (count - expected) * 6, MPI_FLOAT, dstranks[i],
				    TAGBASE_P2 + recv_tags[i], cartcomm, &status) );
	}

	_postrecvC();
    }

    {
	NVTX_RANGE("FSI/halo", NVTX_C7);

	FSI_CORE::setup(wsolvent.p, wsolvent.n, wsolvent.cellsstart, wsolvent.cellscount);

	for(int i = 0; i < 26; ++i)
	    CUDA_CHECK(cudaMemcpyAsync(remote[i].dstate.data, remote[i].hstate.data, sizeof(Particle) * remote[i].hstate.size,
					       cudaMemcpyHostToDevice, uploadstream));

	int nremote_padded = 0;

	{
	    int recvpackcount[26], recvpackstarts_padded[27];

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
	    Particle * recvpackstates[26];

	    for(int i = 0; i < 26; ++i)
		recvpackstates[i] = remote[i].dstate.data;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_CORE::packstates, recvpackstates,
					       sizeof(recvpackstates), 0, cudaMemcpyHostToDevice, uploadstream));
	}

	{
	    Acceleration * packresults[26];

	    for(int i = 0; i < 26; ++i)
		packresults[i] = remote[i].result.devptr;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_CORE::packresults, packresults,
					       sizeof(packresults), 0, cudaMemcpyHostToDevice, uploadstream));
	}

	if (iterationcount)
	    _wait(reqsendA);

	if(nremote_padded)
	{
	    CUDA_CHECK(cudaStreamSynchronize(uploadstream));

	    FSI_CORE::interactions_halo<<< (nremote_padded + 127) / 128, 128, 0, stream>>>
		(nremote_padded, wsolvent.n, (float *)wsolvent.a, local_trunk.get_float());
	}

	for(int i = 0; i < 26; ++i)
	    local[i].update();

	_postrecvP();

	CUDA_CHECK(cudaEventRecord(evAcomputed, stream));
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeFSI::post_a()
{
    if (wsolutes.size() == 0)
	return;

    NVTX_RANGE("FSI/send-a", NVTX_C1);

    CUDA_CHECK(cudaEventSynchronize(evAcomputed));

    reqsendA.resize(26);
    for(int i = 0; i < 26; ++i)
	MPI_CHECK( MPI_Isend(remote[i].result.data, remote[i].result.size * 3, MPI_FLOAT, dstranks[i], TAGBASE_A + i, cartcomm, &reqsendA[i]) );
}

namespace FSI_PUP
{
    __constant__ float * recvbags[26];

    __global__ void unpack(float * const accelerations, const int nparticles)
    {
	const int npack_padded = cpaddedstarts[26];

	assert(blockDim.x * gridDim.x >= 3 * npack_padded && blockDim.x == 128);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	const int pid = gid / 3;

	if (pid >= npack_padded)
	    return;

	const int key9 = 9 * ((int)(pid >= cpaddedstarts[9]) +
			      (int)(pid >= cpaddedstarts[18]));

	const int key3 = 3 * ((int)(pid >= cpaddedstarts[key9 + 3]) +
			      (int)(pid >= cpaddedstarts[key9 + 6]));

	const int key1 =
	    (int)(pid >= cpaddedstarts[key9 + key3 + 1]) +
	    (int)(pid >= cpaddedstarts[key9 + key3 + 2]);

	const int code = key9 + key3 + key1;
	assert(code >= 0 && code < 26);
	assert(pid >= cpaddedstarts[code] && pid < cpaddedstarts[code + 1]);

	const int lpid = pid - cpaddedstarts[code];
	assert(lpid >= 0);

	if (lpid >= ccounts[code])
	    return;

	const int component = gid % 3;

	const int entry = coffsets[code] + lpid;
	assert(entry >= 0 && entry <= ccapacities[code]);
	const float myval = _ACCESS(recvbags[code] + component +  3 * entry);
	const int dpid = _ACCESS(scattered_indices[code] + entry);
	assert(dpid >= 0 && dpid < nparticles);

	atomicAdd(accelerations + 3 * dpid + component, myval);
    }
}

void ComputeFSI::merge_a(cudaStream_t stream)
{
    if (wsolutes.size() == 0)
	return;

    NVTX_RANGE("FSI/merge", NVTX_C2);

    {
	float * recvbags[26];

	for(int i = 0; i < 26; ++i)
	    recvbags[i] = (float *)local[i].result.devptr;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::recvbags, recvbags, sizeof(recvbags), 0, cudaMemcpyHostToDevice, stream));
    }

    _wait(reqrecvA);

    for(int i = 0; i < wsolutes.size(); ++i)
    {
	const ParticlesWrap it = wsolutes[i];

	CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::cpaddedstarts, packsstart.data + 27 * i, sizeof(int) * 27, 0, cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::ccounts, packscount.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemcpyToSymbolAsync(FSI_PUP::coffsets, packsoffset.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));

	if (it.n)
	    FSI_PUP::unpack<<< ((it.n + 31 * 26) * 3 + 127) / 128, 128, 0, stream >>>((float *)it.a, it.n);
    }

    wsolutes.clear();
}

ComputeFSI::~ComputeFSI()
{
    MPI_CHECK(MPI_Comm_free(&cartcomm));

    CUDA_CHECK(cudaEventDestroy(evPpacked));
    CUDA_CHECK(cudaEventDestroy(evAcomputed));
}
