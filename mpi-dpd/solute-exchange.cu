/*
 *  solute-exchange.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

//#define _DUMBCRAY_

#include <../dpd-rng.h>

#include "common-kernels.h"
#include "solute-exchange.h"

namespace SolutePUP
{
    __constant__ int ccapacities[26], * scattered_indices[26];
}

SoluteExchange::SoluteExchange(MPI_Comm _cartcomm) :
iterationcount(-1), packstotalstart(27), host_packstotalstart(27), host_packstotalcount(26)
{
    assert(XSIZE_SUBDOMAIN % 2 == 0 && YSIZE_SUBDOMAIN % 2 == 0 && ZSIZE_SUBDOMAIN % 2 == 0);
    assert(XSIZE_SUBDOMAIN >= 4 && YSIZE_SUBDOMAIN >= 4 && ZSIZE_SUBDOMAIN >= 4);

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));

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

	CUDA_CHECK(cudaMemcpyToSymbol(SolutePUP::ccapacities, &local[i].scattered_indices.capacity, sizeof(int),
				      sizeof(int) * i, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyToSymbol(SolutePUP::scattered_indices, &local[i].scattered_indices.data,
				      sizeof(int *), sizeof(int *) * i, cudaMemcpyHostToDevice));
    }

    _adjust_packbuffers();

    CUDA_CHECK(cudaEventCreateWithFlags(&evPpacked, cudaEventDisableTiming | cudaEventBlockingSync));
    CUDA_CHECK(cudaEventCreateWithFlags(&evAcomputed, cudaEventDisableTiming | cudaEventBlockingSync));

    CUDA_CHECK(cudaPeekAtLastError());
}

namespace SolutePUP
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

	if (lane >= nsrc)
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

	assert(fabs(s1.y) < 1e4);
	assert(fabs(s2.x) < 1e4);
	assert(fabs(s2.y) < 1e4);

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

		if (newcount > ccapacities[tid])
		    failed = true;
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

	const int warpid = threadIdx.x >> 5;
	const int npack_padded = cpaddedstarts[26];

	for (int localbase = 32 * (warpid + 4 * blockIdx.x); localbase < npack_padded; localbase += gridDim.x * blockDim.x)
	{
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
		const int entry = coffsets[code] + packbase + lane;
		assert(entry >= 0 && entry < ccapacities[code]);
		const int pid = _ACCESS(scattered_indices[code] + entry);
		assert(pid >= 0 && pid < nparticles);

		const int entry2 = 3 * pid;

		s0 = _ACCESS(particles + entry2);
		s1 = _ACCESS(particles + entry2 + 1);
		s2 = _ACCESS(particles + entry2 + 2);

		s0.x -= ((code + 2) % 3 - 1) * XSIZE_SUBDOMAIN;
		s0.y -= ((code / 3 + 2) % 3 - 1) * YSIZE_SUBDOMAIN;
		s1.x -= ((code / 9 + 2) % 3 - 1) * ZSIZE_SUBDOMAIN;
	    }

	    assert(cbases[code] + coffsets[code] + packbase >= 0 && npack >= 0);
	    assert(cbases[code] + coffsets[code] + packbase + npack < nbuffer);

	    write_AOS6f(buffer + 3 * (cbases[code] + coffsets[code] + packbase), npack, s0, s1, s2);
	}
    }
}

void SoluteExchange::_pack_attempt(cudaStream_t stream)
{
#ifndef NDEBUG
    CUDA_CHECK(cudaMemsetAsync(packbuf.data, 0xff, sizeof(Particle) * packbuf.capacity, stream));
    memset(host_packbuf.data, 0xff, sizeof(Particle) * packbuf.capacity);

    for(int i = 0; i < 26; ++i)
    {
	CUDA_CHECK(cudaMemsetAsync(local[i].scattered_indices.data, 0x8f, sizeof(int) * local[i].scattered_indices.capacity, stream));
	CUDA_CHECK(cudaMemsetAsync(local[i].result.data, 0xff, sizeof(Acceleration) * local[i].result.capacity, stream));
    }
#endif
    CUDA_CHECK(cudaPeekAtLastError());

    if (packscount.size)
    CUDA_CHECK(cudaMemsetAsync(packscount.data, 0, sizeof(int) * packscount.size, stream));

    if (packsoffset.size)
    CUDA_CHECK(cudaMemsetAsync(packsoffset.data, 0, sizeof(int) * packsoffset.size, stream));

    if (packsstart.size)
    CUDA_CHECK(cudaMemsetAsync(packsstart.data, 0, sizeof(int) * packsstart.size, stream));

    SolutePUP::init<<< 1, 1, 0, stream >>>();

    for(int i = 0; i < wsolutes.size(); ++i)
    {
	const ParticlesWrap it = wsolutes[i];

    if (it.n)
	{
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::coffsets, packsoffset.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));

	    SolutePUP::scatter_indices<<< (it.n + 127) / 128, 128, 0, stream >>>((float2 *)it.p, it.n, packscount.data + i * 26);
	}

	SolutePUP::tiny_scan<<< 1, 32, 0, stream >>>(packscount.data + i * 26, packsoffset.data + 26 * i,
						   packsoffset.data + 26 * (i + 1), packsstart.data + i * 27);

	CUDA_CHECK(cudaPeekAtLastError());
    }

    CUDA_CHECK(cudaMemcpyAsync(host_packstotalcount.data, packsoffset.data + 26 * wsolutes.size(), sizeof(int) * 26, cudaMemcpyDeviceToHost, stream));

    SolutePUP::tiny_scan<<< 1, 32, 0, stream >>>(packsoffset.data + 26 * wsolutes.size(), NULL, NULL, packstotalstart.data);

    CUDA_CHECK(cudaMemcpyAsync(host_packstotalstart.data, packstotalstart.data, sizeof(int) * 27, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::cbases, packstotalstart.data, sizeof(int) * 27, 0, cudaMemcpyDeviceToDevice, stream));

    for(int i = 0; i < wsolutes.size(); ++i)
    {
	const ParticlesWrap it = wsolutes[i];

	if (it.n)
	{
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::coffsets, packsoffset.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::ccounts, packscount.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::cpaddedstarts, packsstart.data + 27 * i, sizeof(int) * 27, 0, cudaMemcpyDeviceToDevice, stream));

	    SolutePUP::pack<<< 14 * 16, 128, 0, stream >>>((float2 *)it.p, it.n, (float2 *)packbuf.data, packbuf.capacity, i);
	}
    }

    CUDA_CHECK(cudaEventRecord(evPpacked, stream));

    CUDA_CHECK(cudaPeekAtLastError());
}

void SoluteExchange::pack_p(cudaStream_t stream)
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

void SoluteExchange::post_p(cudaStream_t stream, cudaStream_t downloadstream)
{
    if (wsolutes.size() == 0)
	return;

    CUDA_CHECK(cudaPeekAtLastError());

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

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::ccapacities, newcapacities, sizeof(newcapacities), 0, cudaMemcpyHostToDevice, stream));

	    int * newindices[26];
	    for(int i = 0; i < 26; ++i)
		newindices[i] = local[i].scattered_indices.data;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::scattered_indices, newindices, sizeof(newindices), 0, cudaMemcpyHostToDevice, stream));

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

	for(int i = 0; i < 26; ++i)
	    local[i].resize(send_counts[i]);

	_postrecvA();

	if (iterationcount == 0)
	{
#ifndef _DUMBCRAY_
	    _postrecvP();
#endif
	}
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

	    _not_nan((float *)(host_packbuf.data + start), count * 6);

#ifdef _DUMBCRAY_
	    MPI_CHECK( MPI_Isend(host_packbuf.data + start, count * 6, MPI_FLOAT, dstranks[i], TAGBASE_P + i, cartcomm, &reqP) );
#else
	    MPI_CHECK( MPI_Isend(host_packbuf.data + start, expected * 6, MPI_FLOAT, dstranks[i], TAGBASE_P + i, cartcomm, &reqP) );
#endif

	    reqsendP.push_back(reqP);

#ifndef _DUMBCRAY_
	    if (count > expected)
	    {
		MPI_Request reqP2;
		MPI_CHECK( MPI_Isend(host_packbuf.data + start + expected, (count - expected) * 6,
				     MPI_FLOAT, dstranks[i], TAGBASE_P2 + i, cartcomm, &reqP2) );

		reqsendP.push_back(reqP2);
	    }
#endif
	}
    }
}

void SoluteExchange::recv_p(cudaStream_t uploadstream)
{
    if (wsolutes.size() == 0)
	return;

    NVTX_RANGE("FSI/recv-p", NVTX_C7);
    
    _wait(reqrecvC);
    _wait(reqrecvP);

    for(int i = 0; i < 26; ++i)
    {
	const int count = recv_counts[i];
	const int expected = remote[i].expected();

	remote[i].pmessage.resize(max(1, count));
	remote[i].preserve_resize(count);

#ifndef NDEBUG
	CUDA_CHECK(cudaMemsetAsync(remote[i].dstate.data, 0xff, sizeof(Particle) * remote[i].dstate.capacity, uploadstream));
	CUDA_CHECK(cudaMemsetAsync(remote[i].result.data, 0xff, sizeof(Acceleration) * remote[i].result.capacity, uploadstream));
#endif

	MPI_Status status;

#ifdef _DUMBCRAY_
	MPI_CHECK( MPI_Recv(remote[i].hstate.data, count * 6, MPI_FLOAT, dstranks[i], TAGBASE_P + recv_tags[i], cartcomm, &status) );
#else
	if (count > expected)
	    MPI_CHECK( MPI_Recv(&remote[i].pmessage.front() + expected, (count - expected) * 6, MPI_FLOAT, dstranks[i],
				TAGBASE_P2 + recv_tags[i], cartcomm, &status) );

	memcpy(remote[i].hstate.data, &remote[i].pmessage.front(), sizeof(Particle) * count);
#endif

	_not_nan((float*)remote[i].hstate.data, count * 6);
    }

    _postrecvC();
    
    for(int i = 0; i < 26; ++i)
	CUDA_CHECK(cudaMemcpyAsync(remote[i].dstate.data, remote[i].hstate.data, sizeof(Particle) * remote[i].hstate.size,
				   cudaMemcpyHostToDevice, uploadstream));
}

void SoluteExchange::halo(cudaStream_t uploadstream, cudaStream_t stream)
{
    NVTX_RANGE("FSI/halo", NVTX_C7);

    if (wsolutes.size() == 0)
	return;
        
    if (iterationcount)
	_wait(reqsendA);
    
    ParticlesWrap halos[26];
    
    for(int i = 0; i < 26; ++i)
	halos[i] = ParticlesWrap(remote[i].dstate.data, remote[i].dstate.size, remote[i].result.devptr);
    
    CUDA_CHECK(cudaStreamSynchronize(uploadstream));
    
    for(int i = 0; i < visitors.size(); ++i)
	visitors[i]->halo(halos, stream);
    
    CUDA_CHECK(cudaPeekAtLastError());
    
    CUDA_CHECK(cudaEventRecord(evAcomputed, stream));
    
    for(int i = 0; i < 26; ++i)
	local[i].update();
    
#ifndef _DUMBCRAY_
    _postrecvP();
#endif
}

void SoluteExchange::post_a()
{
    if (wsolutes.size() == 0)
	return;

    NVTX_RANGE("FSI/send-a", NVTX_C1);

    CUDA_CHECK(cudaEventSynchronize(evAcomputed));

    reqsendA.resize(26);
    for(int i = 0; i < 26; ++i)
	MPI_CHECK( MPI_Isend(remote[i].result.data, remote[i].result.size * 3, MPI_FLOAT, dstranks[i], TAGBASE_A + i, cartcomm, &reqsendA[i]) );
}

namespace SolutePUP
{
    __constant__ float * recvbags[26];

    __global__ void unpack(float * const accelerations, const int nparticles)
    {
	const int npack_padded = cpaddedstarts[26];

	for(int gid = threadIdx.x + blockDim.x * blockIdx.x; gid < 3 * npack_padded; gid += blockDim.x * gridDim.x)
	{
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
		continue;

	    const int component = gid % 3;

	    const int entry = coffsets[code] + lpid;
	    assert(entry >= 0 && entry <= ccapacities[code]);

	    const float myval = _ACCESS(recvbags[code] + component +  3 * entry);
	    const int dpid = _ACCESS(scattered_indices[code] + entry);
	    assert(dpid >= 0 && dpid < nparticles);

	    atomicAdd(accelerations + 3 * dpid + component, myval);
	}
    }
}

void SoluteExchange::recv_a(cudaStream_t stream)
{
    CUDA_CHECK(cudaPeekAtLastError());

    if (wsolutes.size() == 0)
	return;

    NVTX_RANGE("FSI/merge", NVTX_C2);

    {
	float * recvbags[26];

	for(int i = 0; i < 26; ++i)
	    recvbags[i] = (float *)local[i].result.devptr;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::recvbags, recvbags, sizeof(recvbags), 0, cudaMemcpyHostToDevice, stream));
    }

    _wait(reqrecvA);

    for(int i = 0; i < wsolutes.size(); ++i)
    {
	const ParticlesWrap it = wsolutes[i];

	if (it.n)
	{
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::cpaddedstarts, packsstart.data + 27 * i, sizeof(int) * 27, 0, cudaMemcpyDeviceToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::ccounts, packscount.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(SolutePUP::coffsets, packsoffset.data + 26 * i, sizeof(int) * 26, 0, cudaMemcpyDeviceToDevice, stream));

	    SolutePUP::unpack<<< 16 * 14, 128, 0, stream >>>((float *)it.a, it.n);
	}
	CUDA_CHECK(cudaPeekAtLastError());
    }
}

SoluteExchange::~SoluteExchange()
{
    MPI_CHECK(MPI_Comm_free(&cartcomm));

    CUDA_CHECK(cudaEventDestroy(evPpacked));
    CUDA_CHECK(cudaEventDestroy(evAcomputed));
}
