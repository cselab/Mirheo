#include <cassert>
#include <vector>
#include <algorithm>

#include "redistribute-particles.h"

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

using namespace std;

namespace RedistributeParticlesKernels
{
    __constant__ RedistributeParticles::PackBuffer pack_buffers[27];
    
    __constant__ RedistributeParticles::UnpackBuffer unpack_buffers[27];
    
    __device__ int pack_count[27], pack_start[28];

    __constant__ int unpack_start[28];

    __device__ bool failed;
    
    texture<float, cudaTextureType1D> texAllParticles;
 
    __global__ void setup()
    {
	if (threadIdx.x == 0)
	    failed = false;
	
	if (threadIdx.x < 27)
	    pack_count[threadIdx.x] = 0;
    }
    
    __global__ void scatter_halo_indices(const int np, bool * const failureflag)
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
		
		if (entry >= pack_buffers[code].capacity)
		{
		    *failureflag = true;
		    
		    failed = true;
		    
		    //printf("failure for pid %d, look, capacity is %d, my entry is %d\n", pid, buffers[code].capacity, entry);
		}
		else
		    pack_buffers[code].scattered_indices[entry] = pid;
	    }
	}
    }

    __global__ void tiny_scan(const int nparticles, int * const packsizes)
    {
	assert(blockDim.x > 27 && gridDim.x == 1);
	
	const int tid = threadIdx.x;

	int myval = 0, mycount = 0;
	
	if (tid < 27)
	    myval = mycount = pack_count[threadIdx.x];

	if (tid < 27)
	    packsizes[tid] = mycount;

	for(int L = 1; L < 32; L <<= 1)
	    myval += (tid >= L) * __shfl_up(myval, L) ;

	pack_start[tid] = myval - mycount;

	if (tid == 26)
	{
	    //printf("halo particles: %d\n", myval);
	    pack_start[tid + 1] = myval;
	    packsizes[0] = nparticles - myval;
	    //printf("bulk size from device %d\n", nparticles - myval);
	}
    }

#ifndef NDEBUG
    __global__ void check_scan()
    {
	assert(blockDim.x == 1 && gridDim.x == 1);

	for(int i = 1; i < 28; ++i)
	    assert(pack_start[i - 1] <= pack_start[i]);
	
	//for(int i = 0; i < 28; ++i)
	//    printf("%d: %d (count %d), capacity %d\n", i, start[i], i < 27 ? count[i] : 0, i < 27 ? buffers[i].capacity : 0);

	//if (failed)
	//    printf("current status is: FAILED\n");
    }
#endif

    __global__ void pack(const int nparticles, const int nfloats)
    {
	assert(blockDim.x * gridDim.x >= nfloats);

	if (failed)
	    return;
	
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	const int slot = gid / 6;

	const int tid = threadIdx.x;
	
	__shared__ int start[28];

	if (tid < 28)
	    start[tid] = pack_start[tid];

	__syncthreads();

	const int key9 = 9 * (slot >= start[9]) + 9 * (slot >= start[18]);
	const int key3 = 3 * (slot >= start[key9 + 3]) + 3 * (slot >= start[key9 + 6]);
	const int key1 = (slot >= start[key9 + key3 + 1]) + (slot >= start[key9 + key3 + 2]);

	const int idpack = key9 + key3 + key1;

	if (slot >= pack_start[27])
	    return;

	const int offset = slot - pack_start[idpack];
	assert (offset >= 0 && offset < pack_buffers[idpack].capacity);
	
	const int pid = pack_buffers[idpack].scattered_indices[offset];
	assert(pid < nparticles && pid >= 0);

	const int c = gid % 6;
	const int d = c + 6 * offset;
	assert (d < pack_buffers[idpack].capacity * 6);	
	   
	pack_buffers[idpack].buffer[d] = tex1Dfetch(texAllParticles, c + 6 * pid);
    }

    __device__ void bitonic_warp(int& key, int& val)
    {
	const int lane = threadIdx.x & (WARPSIZE - 1);

#pragma unroll
	for(int D = 1; D <= 16; D <<= 1)
#pragma unroll
	    for(int L = D; L >= 1; L >>= 1)
	    { 
		const int mask = L == D ? 2 * D - 1 : L;
		
		const int otherkey = __shfl_xor(key, mask);
		const int otherval = __shfl_xor(val, mask);
		
		const bool exchange =  (2 * (int)(lane < (lane ^ mask)) - 1) * (key - otherkey) > 0;
		
		if (exchange)
		{
		    key = otherkey;
		    val = otherval;
		}
	    }
    }

    __device__ int count_warp(int p)
    {
	for(int L = WARPSIZE / 2; L > 0; L >>=1)
	    p += __shfl_xor(p, L);

	return p;
    }
    
    template<int STRIPESIZE, int ILP>
    __global__ void recompact_bulk(const int np)
    {
	assert(STRIPESIZE == blockDim.x);
	assert(WARPSIZE == warpSize);
	assert(STRIPESIZE % WARPSIZE == 0);

	const int tid = threadIdx.x;
	const int lid = threadIdx.x & (WARPSIZE - 1);
	const int gid = threadIdx.x + STRIPESIZE * blockIdx.x;
     
	int tagged = gid >= np;

	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	if (gid < np)
	    for(int c = 0; c < 3; ++c)
	    {
		const float val = tex1Dfetch(texAllParticles, c + 6 * gid);
		tagged += (int)(val < -L[c] / 2 || val >= L[c] / 2);
	    }
			
	__shared__ int global_offset, local_offset, values[STRIPESIZE];

	if (tid == 0)
	    local_offset = 0;	   

	const int ntags = __syncthreads_count(tagged > 0);
	const int nvalid = STRIPESIZE - ntags;

	if (tid == 0)
	    global_offset = atomicAdd(&pack_count[0], nvalid);
	
	if (ntags)
	{
	    int pid = gid;
	    bitonic_warp(tagged, pid);
	    const int ngoodones = count_warp(!tagged);
	    
	    int warp_offset;

	    if (lid == 0)
		warp_offset = atomicAdd(&local_offset, ngoodones);

	    warp_offset = __shfl(warp_offset, 0);
		
	    if (!tagged)
		values[warp_offset + lid] = pid;

	    assert(warp_offset + lid < nvalid && warp_offset + lid >= 0 || tagged);

	    __syncthreads();

	    const int start = 6 * global_offset;
	    const int stop = start + 6 * nvalid;
	    	    
	    for(int dbase = start + tid; dbase < stop; dbase += STRIPESIZE * ILP)
	    {
		float data[ILP];
#pragma unroll
		for(int i = 0; i < ILP; ++i)
		{
		    const int d = dbase + i * STRIPESIZE;
		    const int c = d % 6;
		    const int s =  (d - start) / 6;
		    assert(s >= 0);
		    assert(s < STRIPESIZE || d >= stop);
		    data[i] = d < stop ? tex1Dfetch(texAllParticles, c + 6 * values[s]) : 0;
		}
#pragma unroll
		for(int i = 0; i < ILP; ++i)
		{
		    const int d = dbase + i * STRIPESIZE;
		    assert(d < pack_buffers[0].capacity * 6);
		    assert(d >= 0);
		    
		    if (d < stop)
			pack_buffers[0].buffer[d] = data[i];
		}
	    }
	}
	else
	{
	    __syncthreads();
	    
	    const int start = 6 * global_offset;
	    const int stop = start + 6 * nvalid;
	    const int srcbase = -start + 6 * STRIPESIZE * blockIdx.x;
	    
	    for(int d = start + tid; d < stop; d += STRIPESIZE * ILP)
	    {
		float data[ILP];

#pragma unroll
		for(int i = 0; i < ILP; ++i)
		{
		    const int s = d + i * STRIPESIZE;
		    data[i] = s < stop ? tex1Dfetch(texAllParticles, srcbase + s) : 0;
		}

#pragma unroll
		for(int i = 0; i < ILP; ++i)
		{
		    const int dest = d + i * STRIPESIZE;
		    if (dest < stop)
		    {
			assert(dest < pack_buffers[0].capacity * 6);
			assert(dest >= 0);
			pack_buffers[0].buffer[dest] = data[i];
		    }
		}
	    }
	}
    }

    __global__ void unpack(float * dstbuf, const int nfloats, const int nparticles, const int base)
    {
	assert(blockDim.x * gridDim.x >= nfloats);
	
	const int gid = threadIdx.x + blockDim.x * blockIdx.x + base;

	if (gid >= nfloats + base)
	    return;
	
	const int slot = gid / 6;
	
	const int key9 = 9 * (slot >= unpack_start[9]) + 9 * (slot >= unpack_start[18]);
	const int key3 = 3 * (slot >= unpack_start[key9 + 3]) + 3 * (slot >= unpack_start[key9 + 6]);
	const int key1 = (slot >= unpack_start[key9 + key3 + 1]) + (slot >= unpack_start[key9 + key3 + 2]);
	const int code = key9 + key3 + key1;
	
	assert(slot >= unpack_start[code] && slot < unpack_start[code + 1]);
	
	const int offset = slot - unpack_start[code];
	assert (offset >= 0 && offset < unpack_buffers[code].capacity);
	
	const int c = gid % 6;
	assert(c >= 0 && c < 6);

	const int s = c + 6 * offset;
	assert (s < unpack_buffers[code].capacity * 6);
	const float value = unpack_buffers[code].buffer[s];
	
	const int shift =
	    XSIZE_SUBDOMAIN * (c == 0) * ((code + 1) % 3 - 1) +
	    YSIZE_SUBDOMAIN * (c == 1) * ((code / 3 + 1) % 3 - 1) +
	    ZSIZE_SUBDOMAIN * (c == 2) * ((code / 9 + 1) % 3 - 1);

	dstbuf[gid] = value + shift;

	//if (!(c >= 3 || fabs(dstbuf[gid]) <= L /2))
	//    printf("error! pid %d c %d code %d x: %f = original %f + shift %f\n", slot, c, code, dstbuf[gid], old, shift);
#ifndef NDEBUG
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	assert(c >= 3 || fabs(dstbuf[gid]) <= L[c] /2);
#endif
    }

#ifndef NDEBUG
    __global__ void check(const Particle * const p, const int np)
    {
	assert(blockDim.x * gridDim.x >= np);

	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };	

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid < np)
	    for(int c = 0; c < 3; ++c)
	    {
		if (!(p[pid].x[c] >= -L[c]/2 && p[pid].x[c] < L[c]/2))
		{
		     printf("oooops pid %d component %d is %f\n", pid, c, p[pid].x[c]);
		}
		
		assert(p[pid].x[c] >= -L[c]/2 && p[pid].x[c] < L[c]/2);
	    }
    }
#endif
}

RedistributeParticles::RedistributeParticles(MPI_Comm _cartcomm): 
failure(1), packsizes(27), nactiveneighbors(26), firstcall(true)
{
    safety_factor = getenv("RDP_COMM_FACTOR") ? atof(getenv("RDP_COMM_FACTOR")) : 1.2;

    MPI_CHECK(MPI_Comm_dup(_cartcomm, &cartcomm) );

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
	
	if (i)
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

    CUDA_CHECK(cudaEventCreate(&evpacking, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&evsizes, cudaEventDisableTiming));
    //CUDA_CHECK(cudaEventCreate(&evcompaction, cudaEventDisableTiming));
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

void RedistributeParticles::_adjust_recv_buffers(const int requested_capacities[27])
{
    for(int i = 0; i < 27; ++i)
    {
	if (requested_capacities[i] <= unpackbuffers[i].capacity)
	    continue;

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
	    printf("ooooooooooooooops %d , req %d!!\n", unpackbuffers[i].capacity, capacity);
	    exit(-1);
	    CUDA_CHECK(cudaFree(unpackbuffers[i].buffer));
	    
	    CUDA_CHECK(cudaMalloc(&unpackbuffers[i].buffer, sizeof(float) * 6 * capacity));
	    assert(pinnedhost_recvbufs[i] == NULL);
	}
	
	unpackbuffers[i].capacity = capacity;
    }
}

int RedistributeParticles::stage1(const Particle * const particles, const int nparticles, cudaStream_t mystream)
{
    NVTX_RANGE("RDP/stage1");
    
    if (firstcall)
	_post_recv();
      
    size_t textureoffset;
    CUDA_CHECK(cudaBindTexture(&textureoffset, &RedistributeParticlesKernels::texAllParticles, particles, 
			       &RedistributeParticlesKernels::texAllParticles.channelDesc,
			       sizeof(float) * 6 * nparticles));
pack_attempt:
    
    CUDA_CHECK(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::pack_buffers, packbuffers,
				       sizeof(PackBuffer) * 27, 0, cudaMemcpyHostToDevice, mystream));

    *failure.data = false;

    RedistributeParticlesKernels::setup<<<1, 32, 0, mystream>>>();

    if (nparticles)
	RedistributeParticlesKernels::scatter_halo_indices<<< (nparticles + 127) / 128, 128, 0, mystream>>>(nparticles, failure.devptr);
    
    RedistributeParticlesKernels::tiny_scan<<<1, 32, 0, mystream>>>(nparticles, packsizes.devptr);

    CUDA_CHECK(cudaEventRecord(evsizes));
    
#ifndef NDEBUG
    RedistributeParticlesKernels::check_scan<<<1, 1, 0, mystream>>>();
#endif 
    
    if (nparticles)
	RedistributeParticlesKernels::pack<<< (6 * nparticles + 127) / 128, 128, 0, mystream>>> (nparticles, nparticles * 6);

    CUDA_CHECK(cudaEventRecord(evpacking));
    
    CUDA_CHECK(cudaEventSynchronize(evsizes));
        
    if (*failure.data)
    {
	//wait for packing to finish
	CUDA_CHECK(cudaEventSynchronize(evpacking));

	printf("...FAILED! Recovering now...\n");

	_adjust_send_buffers(packsizes.devptr);

	goto pack_attempt;
    }

    CUDA_CHECK(cudaPeekAtLastError());

    //CUDA_CHECK(cudaMemset(packbuffers[0].buffer, 0xff, sizeof(float) * 6 * packbuffers[0].capacity));
    
    enum { BS = 128, ILP = 2 };

    if (nparticles)
	RedistributeParticlesKernels::recompact_bulk<BS, ILP><<< (nparticles + BS - 1) / BS, BS, 0, mystream>>>(nparticles);

    //CUDA_CHECK(cudaEventRecord(evcompaction));

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
	if (send_sizes[i] > default_message_sizes[i])
	{
	    const int count = send_sizes[i] - default_message_sizes[i];
	    
	    MPI_CHECK( MPI_Isend(pinnedhost_sendbufs[i] + default_message_sizes[i] * 6, count * 6, MPI_FLOAT,
				 neighbor_ranks[i], basetag + i + 666, cartcomm, sendmsgreq + nsendmsgreq) );
	    ++nsendmsgreq;
	}
    assert(nactiveneighbors <= nsendmsgreq && nsendmsgreq <= 2 * nactiveneighbors);
    
    _waitall(recvcountreq, nactiveneighbors);

    {
	int ustart[28];
	
	ustart[0] = 0;	
	for(int i = 1; i < 28; ++i)
	    ustart[i] = ustart[i - 1] + recv_sizes[i - 1];
	
	CUDA_CHECK(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::unpack_start, ustart,
					   sizeof(int) * 28, 0, cudaMemcpyHostToDevice, mystream));
    }

    nexpected = 0;
    for(int i = 0; i < 27; ++i)
	nexpected += recv_sizes[i];

    nhalo = nexpected - nbulk;
    
    //CUDA_CHECK(cudaEventSynchronize(evcompaction));

    firstcall = false;
    
    return nexpected;
}
    
void RedistributeParticles::stage2(Particle * const particles, const int nparticles, cudaStream_t mystream)
{
    NVTX_RANGE("RDP/stage2");
    
    assert(nparticles == nexpected);
    
    _waitall(recvmsgreq, nactiveneighbors);
    
    _adjust_recv_buffers(recv_sizes);

    CUDA_CHECK(cudaMemcpyToSymbolAsync(RedistributeParticlesKernels::unpack_buffers, unpackbuffers,
				       sizeof(UnpackBuffer) * 27, 0, cudaMemcpyHostToDevice, mystream));
    
    for(int i = 1; i < 27; ++i)
	if (recv_sizes[i] > default_message_sizes[i])
	{
	    const int count = recv_sizes[i] - default_message_sizes[i];
	    
	    MPI_Status status;
	    MPI_CHECK( MPI_Recv(pinnedhost_recvbufs[i] + default_message_sizes[i] * 6, count * 6, MPI_FLOAT,
				neighbor_ranks[i], basetag + recv_tags[i] + 666, cartcomm, &status) );
	}

    CUDA_CHECK(cudaMemcpyAsync(particles, packbuffers[0].buffer, sizeof(Particle) * nbulk, cudaMemcpyDeviceToDevice, mystream));

    if (nhalo)
	RedistributeParticlesKernels::unpack<<<(nhalo * 6 + 127) / 128, 128, 0, mystream>>>((float *)particles, nhalo * 6,
										 nhalo, nbulk * 6);	
    
#ifndef NDEBUG
    RedistributeParticlesKernels::check<<<(nparticles + 127) / 128, 128, 0, mystream>>>(particles, nparticles);
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

