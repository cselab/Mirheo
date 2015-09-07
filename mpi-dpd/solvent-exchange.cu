/*
 *  halo-exchanger.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-05.
 *  Major editing from Massimo-Bernaschi on 2015-03-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstring>
#include <algorithm>

#include "solvent-exchange.h"

using namespace std;

SolventExchange::SolventExchange(MPI_Comm _cartcomm, const int basetag):  basetag(basetag), firstpost(true), nactive(26)
{
    safety_factor = getenv("HEX_COMM_FACTOR") ? atof(getenv("HEX_COMM_FACTOR")) : 1.2;

    assert(XSIZE_SUBDOMAIN % 2 == 0 && YSIZE_SUBDOMAIN % 2 == 0 && ZSIZE_SUBDOMAIN % 2 == 0);
    assert(XSIZE_SUBDOMAIN >= 2 && YSIZE_SUBDOMAIN >= 2 && ZSIZE_SUBDOMAIN >= 2);

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
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

	halosize[i].x = d[0] != 0 ? 1 : XSIZE_SUBDOMAIN;
	halosize[i].y = d[1] != 0 ? 1 : YSIZE_SUBDOMAIN;
	halosize[i].z = d[2] != 0 ? 1 : ZSIZE_SUBDOMAIN;

	const int nhalocells = halosize[i].x * halosize[i].y * halosize[i].z;

	int estimate = numberdensity * safety_factor * nhalocells;
	estimate = 32 * ((estimate + 31) / 32);

	recvhalos[i].setup(estimate, nhalocells);
	sendhalos[i].setup(estimate, nhalocells);
    }

    CUDA_CHECK(cudaHostAlloc((void **)&required_send_bag_size_host, sizeof(int) * 26, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&required_send_bag_size, required_send_bag_size_host, 0));

    CUDA_CHECK(cudaEventCreateWithFlags(&evfillall, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evdownloaded, cudaEventDisableTiming | cudaEventBlockingSync));
}

namespace PackingHalo
{
    int ncells;

    __constant__ int cellpackstarts[27];

    struct CellPackSOA { int * start, * count, * scan, size; bool enabled; };

    __constant__ CellPackSOA cellpacks[26];

    __global__ void count_all(const int * const cellsstart, const int * const cellscount, const int ntotalcells)
    {
	assert(blockDim.x * gridDim.x >= ntotalcells);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= cellpackstarts[26])
	    return;

	const int key9 = 9 * ((gid >= cellpackstarts[9]) + (gid >= cellpackstarts[18]));
	const int key3 = 3 * ((gid >= cellpackstarts[key9 + 3]) + (gid >= cellpackstarts[key9 + 6]));
	const int key1 = (gid >= cellpackstarts[key9 + key3 + 1]) + (gid >= cellpackstarts[key9 + key3 + 2]);
	const int code = key9 + key3 + key1;

	assert(code >= 0 && code < 26);
	assert(gid >= cellpackstarts[code] && gid < cellpackstarts[code + 1]);

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	int halo_start[3];
	for(int c = 0; c < 3; ++c)
	    halo_start[c] = max(d[c] * L[c] - L[c]/2 - 1, -L[c]/2);

	int halo_size[3];
	for(int c = 0; c < 3; ++c)
	    halo_size[c] = min(d[c] * L[c] + L[c]/2 + 1, L[c]/2) - halo_start[c];

	const int ndstcells = halo_size[0] * halo_size[1] * halo_size[2];

	assert(cellpackstarts[code + 1] - cellpackstarts[code] == ndstcells + 1);

	const int dstcid = gid - cellpackstarts[code];

	if (dstcid < ndstcells)
	{
	    const int dstcellpos[3] = {
		dstcid % halo_size[0],
		(dstcid / halo_size[0]) % halo_size[1],
		dstcid / (halo_size[0] * halo_size[1])
	    };

	    int srccellpos[3];
	    for(int c = 0; c < 3; ++c)
		srccellpos[c] = halo_start[c] + dstcellpos[c] + L[c] / 2;

	    for(int c = 0; c < 3; ++c)
		assert(srccellpos[c] >= 0);

	    for(int c = 0; c < 3; ++c)
		assert(srccellpos[c] < L[c]);

	    const int srcentry = srccellpos[0] + XSIZE_SUBDOMAIN * (srccellpos[1] + YSIZE_SUBDOMAIN * srccellpos[2]);
	    assert(srcentry < XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN);

	    const int enabled = cellpacks[code].enabled;

	    cellpacks[code].start[dstcid] = enabled * cellsstart[srcentry];
	    cellpacks[code].count[dstcid] = enabled * cellscount[srcentry];
	}
	else if (dstcid == ndstcells)
	{
	    cellpacks[code].start[dstcid] = 0;
	    cellpacks[code].count[dstcid] = 0;
	}
    }

    __constant__ int * srccells[26 * 2], * dstcells[26 * 2];

    template<int slot>
    __global__ void copycells(const int n)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= cellpackstarts[26])
	    return;

	const int key9 = 9 * ((gid >= cellpackstarts[9]) + (gid >= cellpackstarts[18]));
	const int key3 = 3 * ((gid >= cellpackstarts[key9 + 3]) + (gid >= cellpackstarts[key9 + 6]));
	const int key1 = (gid >= cellpackstarts[key9 + key3 + 1]) + (gid >= cellpackstarts[key9 + key3 + 2]);

	const int idpack = key9 + key3 + key1;

	const int offset = gid - cellpackstarts[idpack];

	dstcells[idpack + 26 * slot][offset] = srccells[idpack + 26 * slot][offset];
    }

#ifndef NDEBUG
    __device__ void halo_particle_check(const Particle p, const int pid, const int code)
    {
	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	for(int c = 0; c < 3; ++c)
	{
	    const float halo_start = max(d[c] * L[c] - L[c]/2 - 1, -L[c]/2);
	    const float halo_end = min(d[c] * L[c] + L[c]/2 + 1, L[c]/2);
	    const float eps = 1e-5;
	    if (!(p.x[c] >= halo_start - eps && p.x[c] < halo_end + eps))
	    {
		printf("fill particles (pack) oooops particle %d: %e %e %e component %d not within %f , %f eps %e\n", pid,
		       p.x[0], p.x[1], p.x[2], c, halo_start, halo_end, eps);

	    }

	    assert(p.x[c] >= halo_start - eps && p.x[c] < halo_end + eps);
	}
    }
#endif

    template<int NWARPS>
    __global__ void scan_diego()
    {
	assert(gridDim.x == 26 && blockDim.x == 32 * NWARPS);

	__shared__ int shdata[32];

	const int code = blockIdx.x;
	const int * const count = cellpacks[code].count;
	int * const start = cellpacks[code].scan;
	const int n = cellpacks[code].size;

	const int tid = threadIdx.x;
	const int laneid = threadIdx.x & 0x1f;
	const int warpid = threadIdx.x >> 5;

	int lastval = 0;

	for(int sourcebase = 0; sourcebase < n; sourcebase += 32 * NWARPS)
	{
	    const int sourceid = sourcebase + tid;

	    int mycount = 0, myscan = 0;

	    if (sourceid < n)
		myscan = mycount = count[sourceid];

	    if (tid == 0)
		myscan += lastval;

	    for(int L = 1; L < 32; L <<= 1)
	    {
		const int val = __shfl_up(myscan, L);

		if (laneid >= L)
		    myscan += val;
	    }

	    if (laneid == 31)
		shdata[warpid] = myscan;

	    __syncthreads();

	    if (warpid == 0)
	    {
		int gs = 0;

		if (laneid < NWARPS)
		    gs = shdata[tid];

		for(int L = 1; L < 32; L <<= 1)
		{
		    const int val = __shfl_up(gs, L);

		    if (laneid >= L)
			gs += val;
		}

		shdata[tid] = gs;

		lastval = __shfl(gs, 31);
	    }

	    __syncthreads();

	    if (warpid)
		myscan += shdata[warpid - 1];

	    __syncthreads();

	    if (sourceid < n)
		start[sourceid] = myscan - mycount;
	}
    }

    struct SendBagInfo
    {
	const int * start_src, * count_src, * start_dst;
	int bagsize, * scattered_entries;
	Particle * dbag, * hbag;
    };

    __constant__ SendBagInfo baginfos[26];

    __global__ void fill_all(const Particle * const particles, const int np, int * const required_bag_size)
    {
	assert(sizeof(Particle) == 6 * sizeof(float));
	assert(blockDim.x == warpSize);

	const int gcid = (threadIdx.x >> 4) + 2 * blockIdx.x;

	if (gcid >= cellpackstarts[26])
	    return;

	const int key9 = 9 * ((gcid >= cellpackstarts[9]) + (gcid >= cellpackstarts[18]));
	const int key3 = 3 * ((gcid >= cellpackstarts[key9 + 3]) + (gcid >= cellpackstarts[key9 + 6]));
	const int key1 = (gcid >= cellpackstarts[key9 + key3 + 1]) + (gcid >= cellpackstarts[key9 + key3 + 2]);
	const int code = key9 + key3 + key1;

	assert(code >= 0 && code < 26);
	assert(gcid >= cellpackstarts[code] && gcid < cellpackstarts[code + 1]);

	const int cellid = gcid - cellpackstarts[code];

	const int tid = threadIdx.x & 0xf;

	const int base_src = baginfos[code].start_src[cellid];
	const int base_dst = baginfos[code].start_dst[cellid];
	const int nsrc = min(baginfos[code].count_src[cellid], baginfos[code].bagsize - base_dst);

	const int nfloats = nsrc * 6;
	for(int i = 2 * tid; i < nfloats; i += warpSize)
	{
	    const int lpid = i / 6;
	    const int dpid = base_dst + lpid;
	    const int spid = base_src + lpid;
	    assert(spid < np && spid >= 0);

	    const int c = i % 6;

	    float2 word = *(float2 *)&particles[spid].x[c];
	    *(float2 *)&baginfos[code].dbag[dpid].x[c] = word;
	    //*(float2 *)&baginfos[code].hbag[dpid].x[c] = word;

#ifndef NDEBUG
	    halo_particle_check(particles[spid], spid, code)   ;
#endif
	}

	for(int lpid = tid; lpid < nsrc; lpid += warpSize / 2)
	{
	    const int dpid = base_dst + lpid;
	    const int spid = base_src + lpid;

	    baginfos[code].scattered_entries[dpid] = spid;
	}

	if (gcid + 1 == cellpackstarts[code + 1])
	    required_bag_size[code] = base_dst;
    }


#ifndef NDEBUG
    __global__ void check_send_particles(Particle * p, int n, int code)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	assert(p[pid].x[0] >= -L[0] / 2 || p[pid].x[0] < L[0] / 2 ||
	       p[pid].x[1] >= -L[1] / 2 || p[pid].x[1] < L[1] / 2 ||
	       p[pid].x[2] >= -L[2] / 2 || p[pid].x[2] < L[2] / 2);

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	for(int c = 0; c < 3; ++c)
	{
	    const float halo_start = max(d[c] * L[c] - L[c]/2 - 1, -L[c]/2);
	    const float halo_end = min(d[c] * L[c] + L[c]/2 + 1, L[c]/2);
	    const float eps = 1e-5;
	    if (!(p[pid].x[c] >= halo_start - eps && p[pid].x[c] < halo_end + eps))
		printf("oooops particle %d: %e %e %e component %d not within %f , %f eps %f\n",
		       pid, p[pid].x[0], p[pid].x[1], p[pid].x[2],
		       c, halo_start, halo_end, eps);

	    assert(p[pid].x[c] >= halo_start - eps && p[pid].x[c] < halo_end + eps);
	}
    }
#endif
}

void SolventExchange::_pack_all(const Particle * const p, const int n, const bool update_baginfos, cudaStream_t stream)
{
    if (update_baginfos)
    {
	static PackingHalo::SendBagInfo baginfos[26];

	for(int i = 0; i < 26; ++i)
	{
	    baginfos[i].start_src = sendhalos[i].tmpstart.data;
	    baginfos[i].count_src = sendhalos[i].tmpcount.data;
	    baginfos[i].start_dst = sendhalos[i].dcellstarts.data;
	    baginfos[i].bagsize = sendhalos[i].dbuf.capacity;
	    baginfos[i].scattered_entries = sendhalos[i].scattered_entries.data;
	    baginfos[i].dbag = sendhalos[i].dbuf.data;
	    baginfos[i].hbag = sendhalos[i].hbuf.data;
	}

	CUDA_CHECK(cudaMemcpyToSymbolAsync(PackingHalo::baginfos, baginfos, sizeof(baginfos), 0, cudaMemcpyHostToDevice, stream)); // peh: added stream
    }

    PackingHalo::fill_all<<< (PackingHalo::ncells + 1) / 2, 32, 0, stream>>>(p, n, required_send_bag_size);

    CUDA_CHECK(cudaEventRecord(evfillall, stream));
}

void SolventExchange::pack(const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount, cudaStream_t stream)
{
    CUDA_CHECK(cudaPeekAtLastError());

    nlocal = n;

    NVTX_RANGE("HEX/pack", NVTX_C2);

    if (firstpost)
    {
	{
	    static int cellpackstarts[27];

	    cellpackstarts[0] = 0;
	    for(int i = 0, s = 0; i < 26; ++i)
		cellpackstarts[i + 1] =  (s += sendhalos[i].dcellstarts.size * (sendhalos[i].expected > 0));

	    PackingHalo::ncells = cellpackstarts[26];

	    CUDA_CHECK(cudaMemcpyToSymbol(PackingHalo::cellpackstarts, cellpackstarts,
					       sizeof(cellpackstarts), 0, cudaMemcpyHostToDevice));
	}

	{
	    static PackingHalo::CellPackSOA cellpacks[26];
	    for(int i = 0; i < 26; ++i)
	    {
		cellpacks[i].start = sendhalos[i].tmpstart.data;
		cellpacks[i].count = sendhalos[i].tmpcount.data;
		cellpacks[i].enabled = sendhalos[i].expected > 0;
		cellpacks[i].scan = sendhalos[i].dcellstarts.data;
		cellpacks[i].size = sendhalos[i].dcellstarts.size;
	    }

	    CUDA_CHECK(cudaMemcpyToSymbol(PackingHalo::cellpacks, cellpacks,
					       sizeof(cellpacks), 0, cudaMemcpyHostToDevice));
	}
    }

    PackingHalo::count_all<<<(PackingHalo::ncells + 127) / 128, 128, 0, stream>>>(cellsstart, cellscount, PackingHalo::ncells);

    PackingHalo::scan_diego< 32 ><<< 26, 32 * 32, 0, stream>>>();

    CUDA_CHECK(cudaPeekAtLastError());

    if (firstpost)
	post_expected_recv();
    else
    {
	MPI_Status statuses[26 * 2];
	MPI_CHECK( MPI_Waitall(nactive, sendcellsreq, statuses) );
	MPI_CHECK( MPI_Waitall(nsendreq, sendreq, statuses) );
	MPI_CHECK( MPI_Waitall(nactive, sendcountreq, statuses) );
    }

    if (firstpost)
    {
	{
	    static int * srccells[26];
	    for(int i = 0; i < 26; ++i)
		srccells[i] = sendhalos[i].dcellstarts.data;

	    CUDA_CHECK(cudaMemcpyToSymbol(PackingHalo::srccells, srccells, sizeof(srccells), 0, cudaMemcpyHostToDevice));

	    static int * dstcells[26];
	    for(int i = 0; i < 26; ++i)
		dstcells[i] = sendhalos[i].hcellstarts.devptr;

	    CUDA_CHECK(cudaMemcpyToSymbol(PackingHalo::dstcells, dstcells, sizeof(dstcells), 0, cudaMemcpyHostToDevice));
	}

	{
	    static int * srccells[26];
	    for(int i = 0; i < 26; ++i)
		srccells[i] = recvhalos[i].hcellstarts.devptr;

	    CUDA_CHECK(cudaMemcpyToSymbol(PackingHalo::srccells, srccells, sizeof(srccells), sizeof(srccells), cudaMemcpyHostToDevice));

	    static int * dstcells[26];
	    for(int i = 0; i < 26; ++i)
		dstcells[i] = recvhalos[i].dcellstarts.data;

	    CUDA_CHECK(cudaMemcpyToSymbol(PackingHalo::dstcells, dstcells, sizeof(dstcells), sizeof(dstcells), cudaMemcpyHostToDevice));
	}
    }

    PackingHalo::copycells<0><<< (PackingHalo::ncells + 127) / 128, 128, 0, stream>>>(PackingHalo::ncells);

    _pack_all(p, n, firstpost, stream);

}

void SolventExchange::consolidate_and_post(const Particle * const p, const int n, cudaStream_t stream, cudaStream_t downloadstream)
{
    {
	NVTX_RANGE("HEX/consolidate", NVTX_C2);

	CUDA_CHECK(cudaEventSynchronize(evfillall));

	bool succeeded = true;
	for(int i = 0; i < 26; ++i)
	{
	    const int nrequired = required_send_bag_size_host[i];
	    const bool failed_entry = nrequired > sendhalos[i].dbuf.capacity;// || nrequired > sendhalos[i].hbuf.capacity;

	    if (failed_entry)
	    {
		sendhalos[i].dbuf.resize(nrequired);
		//sendhalos[i].hbuf.resize(nrequired);
		sendhalos[i].scattered_entries.resize(nrequired);
		succeeded = false;
	    }
	}

	if (!succeeded)
	{
	    _pack_all(p, n, true, stream);

	    CUDA_CHECK(cudaEventSynchronize(evfillall));
	}

	for(int i = 0; i < 26; ++i)
	{
	    const int nrequired = required_send_bag_size_host[i];

	    sendhalos[i].dbuf.size = nrequired;

	    //sendhalos[i].hbuf.size = nrequired;
	    sendhalos[i].hbuf.resize(nrequired);
	    assert(nrequired <= sendhalos[i].dbuf.capacity && nrequired <= sendhalos[i].hbuf.capacity);

	    sendhalos[i].scattered_entries.size = nrequired;
	}
    }

    for(int i = 0; i < 26; ++i)
	if (sendhalos[i].hbuf.size)
	    CUDA_CHECK(cudaMemcpyAsync(sendhalos[i].hbuf.data, sendhalos[i].dbuf.data, sizeof(Particle) * sendhalos[i].hbuf.size,
				       cudaMemcpyDeviceToHost, downloadstream));

    //this is commented due to the hanging described below
    //CUDA_CHECK(cudaEventRecord(evdownloaded, downloadstream));

#ifndef NDEBUG
    CUDA_CHECK(cudaStreamSynchronize(0));

    for(int i = 0; i < 26; ++i)
	if (sendhalos[i].expected)
	{
	    const int nd = sendhalos[i].dbuf.size;

	    if (nd > 0)
		PackingHalo::check_send_particles<<<(nd + 127)/ 128, 128, 0, stream>>>(sendhalos[i].dbuf.data, nd, i);
	}

    CUDA_CHECK(cudaStreamSynchronize(0));

    CUDA_CHECK(cudaPeekAtLastError());
#endif

    CUDA_CHECK(cudaStreamSynchronize(downloadstream));
    //this hangs undefinitely on XK7 (please confirm)
    //CUDA_CHECK(cudaEventSynchronize(evdownloaded));

    {
	NVTX_RANGE("HEX/send", NVTX_C2);

	for(int i = 0, c = 0; i < 26; ++i)
	    if (sendhalos[i].expected)
		MPI_CHECK( MPI_Isend(sendhalos[i].hcellstarts.data, sendhalos[i].hcellstarts.size, MPI_INTEGER, dstranks[i],
				     basetag + i + 350, cartcomm, sendcellsreq + c++) );

	for(int i = 0, c = 0; i < 26; ++i)
	    if (sendhalos[i].expected)
		MPI_CHECK( MPI_Isend(&sendhalos[i].hbuf.size, 1, MPI_INTEGER, dstranks[i], basetag +  i + 150, cartcomm, sendcountreq + c++) );

	nsendreq = 0;

	for(int i = 0; i < 26; ++i)
	{
	    const int expected = sendhalos[i].expected;

	    if (expected == 0)
		continue;

	    const int count = sendhalos[i].hbuf.size;

	    MPI_CHECK( MPI_Isend(sendhalos[i].hbuf.data, expected, Particle::datatype(), dstranks[i],
				 basetag +  i, cartcomm, sendreq + nsendreq) );

	    ++nsendreq;

	    if (count > expected)
	    {

		const int difference = count - expected;

		int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };
		printf("extra message from rank %d to rank %d in the direction of %d %d %d! difference %d, expected is %d\n",
		       myrank, dstranks[i], d[0], d[1], d[2], difference, expected);

		MPI_CHECK( MPI_Isend(sendhalos[i].hbuf.data + expected, difference, Particle::datatype(), dstranks[i],
				     basetag + i + 555, cartcomm, sendreq + nsendreq) );

		++nsendreq;
	    }
	}
    }

    firstpost = false;
}

void SolventExchange::post_expected_recv()
{
    NVTX_RANGE("HEX/post irecv", NVTX_C3);

    for(int i = 0, c = 0; i < 26; ++i)
    {
	assert(recvhalos[i].hbuf.capacity >= recvhalos[i].expected);

	if (recvhalos[i].expected)
	    MPI_CHECK( MPI_Irecv(recvhalos[i].hbuf.data, recvhalos[i].expected, Particle::datatype(), dstranks[i],
				 basetag + recv_tags[i], cartcomm, recvreq + c++ ));
    }

    for(int i = 0, c = 0; i < 26; ++i)
	if (recvhalos[i].expected)
	    MPI_CHECK( MPI_Irecv(recvhalos[i].hcellstarts.data, recvhalos[i].hcellstarts.size, MPI_INTEGER, dstranks[i],
				 basetag + recv_tags[i] + 350, cartcomm,  recvcellsreq + c++) );

    for(int i = 0, c = 0; i < 26; ++i)
	if (recvhalos[i].expected)
	    MPI_CHECK( MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i],
				 basetag + recv_tags[i] + 150, cartcomm, recvcountreq + c++) );
	else
	    recv_counts[i] = 0;
}

void SolventExchange::wait_for_messages(cudaStream_t stream, cudaStream_t uploadstream)
{
    NVTX_RANGE("HEX/wait-recv", NVTX_C4);

    CUDA_CHECK(cudaPeekAtLastError());

    {
	MPI_Status statuses[26];

	MPI_CHECK( MPI_Waitall(nactive, recvreq, statuses) );
	MPI_CHECK( MPI_Waitall(nactive, recvcellsreq, statuses) );
	MPI_CHECK( MPI_Waitall(nactive, recvcountreq, statuses) );
    }

    for(int i = 0; i < 26; ++i)
    {
	const int count = recv_counts[i];
	const int expected = recvhalos[i].expected;
	const int difference = count - expected;

	if (count <= expected)
	{
	    recvhalos[i].hbuf.resize(count);
	    recvhalos[i].dbuf.resize(count);
	}
	else
	{
	    printf("RANK %d waiting for RECV-extra message: count %d expected %d (difference %d) from rank %d\n",
		   myrank, count, expected, difference, dstranks[i]);

	    recvhalos[i].hbuf.preserve_resize(count);
	    recvhalos[i].dbuf.resize(count);

	    MPI_Status status;

	    MPI_Recv(recvhalos[i].hbuf.data + expected, difference, Particle::datatype(), dstranks[i],
		     basetag + recv_tags[i] + 555, cartcomm, &status);
	}
    }

    for(int i = 0; i < 26; ++i)
	CUDA_CHECK(cudaMemcpyAsync(recvhalos[i].dbuf.data, recvhalos[i].hbuf.data,
				   sizeof(Particle) * recvhalos[i].hbuf.size, cudaMemcpyHostToDevice, uploadstream));

    for(int i = 0; i < 26; ++i)
	CUDA_CHECK(cudaMemcpyAsync(recvhalos[i].dcellstarts.data, recvhalos[i].hcellstarts.data,
				   sizeof(int) * recvhalos[i].hcellstarts.size, cudaMemcpyHostToDevice, uploadstream));

    CUDA_CHECK(cudaPeekAtLastError());

    post_expected_recv();
}

int SolventExchange::nof_sent_particles()
{
    int s = 0;
    for(int i = 0; i < 26; ++i)
	s += sendhalos[i].hbuf.size;

    return s;
}

void SolventExchange::_cancel_recv()
{
    if (!firstpost)
    {
	{
	    MPI_Status statuses[26 * 2];
	    MPI_CHECK( MPI_Waitall(nactive, sendcellsreq, statuses) );
	    MPI_CHECK( MPI_Waitall(nsendreq, sendreq, statuses) );
	    MPI_CHECK( MPI_Waitall(nactive, sendcountreq, statuses) );
	}

	for(int i = 0; i < nactive; ++i)
	    MPI_CHECK( MPI_Cancel(recvreq + i) );

	for(int i = 0; i < nactive; ++i)
	    MPI_CHECK( MPI_Cancel(recvcellsreq + i) );

	for(int i = 0; i < nactive; ++i)
	    MPI_CHECK( MPI_Cancel(recvcountreq + i) );

	firstpost = true;
    }
}

void SolventExchange::adjust_message_sizes(ExpectedMessageSizes sizes)
{
    _cancel_recv();
    nactive = 0;
    for(int i = 0; i < 26; ++i)
    {
	const int d[3] = { (i + 2) % 3, (i / 3 + 2) % 3, (i / 9 + 2) % 3 };
	const int entry = d[0] + 3 * (d[1] + 3 * d[2]);
	int estimate = sizes.msgsizes[entry] * safety_factor;
	estimate = 64 * ((estimate + 63) / 64);

	//if (estimate)
	//    printf("RANK %d: direction %d %d %d: adjusting msg %d with entry %d  to %d and safety factor is %f\n",
	//	   myrank, d[0] - 1, d[1] - 1, d[2] - 1, i, entry, estimate, safety_factor);

	recvhalos[i].adjust(estimate);
	sendhalos[i].adjust(estimate);

    if (estimate == 0)
        required_send_bag_size_host[i] = 0;

	nactive += (int)(estimate > 0);
    }
}

SolventExchange::~SolventExchange()
{
    CUDA_CHECK(cudaFreeHost(required_send_bag_size));

    MPI_CHECK(MPI_Comm_free(&cartcomm));

    _cancel_recv();

    CUDA_CHECK(cudaEventDestroy(evfillall));
    CUDA_CHECK(cudaEventDestroy(evdownloaded));
}
