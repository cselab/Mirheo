#include <algorithm>

#include <cuda-dpd.h>
#include "dpd-interactions.h"

using namespace std;

void ComputeInteractionsDPD::evaluate(int& saru_tag, Particle * p, int n, Acceleration * a, int * cellsstart, int * cellscount)
{
    dpd_remote_interactions_stage1(p, n);
    
    forces_dpd_cuda_nohost((float *)p, (float *)a, n, 
			   cellsstart, cellscount,
			   1, L, L, L, aij, gammadpd, sigma, 1. / sqrt(dt), saru_tag);

    saru_tag += nranks - myrank;

    dpd_remote_interactions_stage2(p, n, saru_tag, a);
    
    saru_tag += 1 + myrank;  
}

namespace PackingHalo
{
    __device__ int blockcount, global_histo[27], requiredsize;
    __device__ int * scattered_entries, scattered_entries_size;

    __global__ void setup(bool firsttime)
    {
	blockcount = 0;

	if (firsttime)
	    scattered_entries_size = 0;
	
	for(int i = 0; i < 27; ++i)
	    global_histo[i] = 0;
    }

    template< int work >
    __global__ void stage1(int * packs_start, Particle * p, int np, const int L, int * bag_size_required)
    {
	assert(blockDim.x * gridDim.x * work >= np);
	assert(blockDim.x >= 26);
    
	__shared__ int histo[26];

	const int tid = threadIdx.x; 

	if (tid < 26)
	    histo[tid] = 0;

	__syncthreads();

	for(int t = 0; t < work; ++t)
	{
	    const int pid = tid + blockDim.x * (blockIdx.x + gridDim.x * t);

	    if (pid < np)
		for(int i = 0; i < 26; ++i)
		{
		    int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

		    bool halo = true;			
	
		    for(int c = 0; c < 3; ++c)
		    {
			const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
			const float halo_end = min(d[c] * L + L/2 + 1, L/2);
		
			const float x = p[pid].x[c];
		
			halo &= (x >= halo_start && x < halo_end);
		    }

		    if (halo)
			atomicAdd(histo + i, 1);
		}
	}

	__syncthreads();
    
	if (tid < 26 && histo[tid] > 0)
	    atomicAdd(global_histo + tid, histo[tid]);

	if (tid == 0)
	{
	    const int timestamp = atomicAdd(&blockcount, 1);

	    if (timestamp == gridDim.x - 1)
	    {
		blockcount = 0;

		int s = 0, curr;

		for(int i = 0; i < 26; ++i)
		{
		    curr = global_histo[i];
		    global_histo[i] = packs_start[i] = s;
		    s += curr;
		}

		global_histo[26] = packs_start[26] = s;
		requiredsize = s;		
		*bag_size_required = s;

		if (scattered_entries_size < s)
		{
		    if (scattered_entries_size > 0)
			delete [] scattered_entries;
		    
		    scattered_entries = new int[s];
		    assert(scattered_entries != NULL);
		    scattered_entries_size = s;
		}
	    }
	}
    }

    __global__ void stage2(Particle * particles, int np, const int L, Particle * bag, int bagsize)
    {
	if (bagsize < requiredsize)
	    return;
	    
	assert(blockDim.x * gridDim.x >= np);
	assert(blockDim.x >= 26);
    
	__shared__ int histo[26];
	__shared__ int base[26];

	const int tid = threadIdx.x; 

	if (tid < 26)
	    histo[tid] = 0;

	__syncthreads();

	int offset[26];
	for(int i = 0; i < 26; ++i)
	    offset[i] = -1;

	Particle p;
    
	const int pid = tid + blockDim.x * blockIdx.x;

	if (pid < np)
	{
	    p = particles[pid];

	    for(int c = 0; c < 3; ++c)
		assert(p.x[c] >= -L / 2 && p.x[c] < L / 2);
	
	    for(int i = 0; i < 26; ++i)
	    {
		int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

		bool halo = true;			
	
		for(int c = 0; c < 3; ++c)
		{
		    const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
		    const float halo_end = min(d[c] * L + L/2 + 1, L/2);
		
		    const float x = p.x[c];
		
		    halo &= (x >= halo_start && x < halo_end);
		}
   
		if (halo)
		    offset[i] = atomicAdd(histo + i, 1);
	    }
	}
	__syncthreads();
    
	if (tid < 26 && histo[tid] > 0)
	    base[tid] = atomicAdd(global_histo + tid, histo[tid]);

	__syncthreads();

	for(int i = 0; i < 26; ++i)
	    if (offset[i] != -1)
	    {
		const int entry = base[i] + offset[i];
		assert(entry >= 0 && entry < global_histo[26]); 
		
		bag[ entry ] = p; 
		scattered_entries[ entry ] = pid;
	    }
    }

    __global__ void shift_remote_particles(Particle * p, int n, int L, int code)
    {
	assert(blockDim.x * gridDim.x >= n);
	
	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	for(int c = 0; c < 3; ++c)
	    assert(p[pid].x[c] >= -L / 2 && p[pid].x[c] < L / 2);

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
	
	for(int c = 0; c < 3; ++c)
	    p[pid].x[c] += d[c] * L;

#ifndef NDEBUG

	assert(p[pid].x[0] <= -L / 2 || p[pid].x[0] >= L / 2 ||
	       p[pid].x[1] <= -L / 2 || p[pid].x[1] >= L / 2 || 
	       p[pid].x[2] <= -L / 2 || p[pid].x[2] >= L / 2);

	for(int c = 0; c < 3; ++c)
	{
	    const float halo_start = max(d[c] * L - L/2, -L/2 - 1);
	    const float halo_end = min(d[c] * L + L/2, L/2 + 1);

	    assert(p[pid].x[c] >= halo_start && p[pid].x[c] <= halo_end);
	}
	
#endif
    }

    __global__ void check_send_particles(Particle * p, int n, int L, int code)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	assert(p[pid].x[0] >= -L / 2 || p[pid].x[0] < L / 2 ||
	       p[pid].x[1] >= -L / 2 || p[pid].x[1] < L / 2 || 
	       p[pid].x[2] >= -L / 2 || p[pid].x[2] < L / 2);

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
	
	for(int c = 0; c < 3; ++c)
	{
	    const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
	    const float halo_end = min(d[c] * L + L/2 + 1, L/2);

	    assert(p[pid].x[c] >= halo_start && p[pid].x[c] < halo_end);
	}
    }

    __global__ void merge_accelerations(Acceleration * aremote, const int nremote, Acceleration * alocal, const int nlocal,
					Particle * premote, Particle * plocal)
    {
	assert(blockDim.x * gridDim.x >= nremote);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= nremote)
	    return;

        int pid = scattered_entries[ gid ];
	assert(pid >= 0 && pid < nlocal);
	
	Acceleration a = aremote[gid];

	for(int c = 0; c < 3; ++c)
	    atomicAdd(&alocal[pid].a[c], a.a[c]);
	
#ifndef NDEBUG
	for(int c = 0; c < 3; ++c)
	{
       	    if (isnan(a.a[c]))
		printf("oouch pid %d %f\n", pid, a.a[c]);
	    
	    assert(!isnan(a.a[c]));
    }

	Particle p1 = plocal[pid];
	Particle p2 = premote[gid];

	for(int c = 0; c < 3; ++c)
	{
	    assert(p1.x[c] == p2.x[c]);
	    assert(p1.x[c] == p2.x[c]);
	}
#endif
    }
}

#include <numeric>
void ComputeInteractionsDPD::dpd_remote_interactions_stage1(Particle * p, int n)
{
    MPI_Status statuses[26];
    
    if (pending_send)
	MPI_CHECK( MPI_Waitall(26, sendreq, statuses) );

    PackingHalo::setup<<<1, 1>>>(false);

    if (n > 0)
	PackingHalo::stage1<1> <<<(n + 127) / 128, 128>>> (sendpacks_start, p, n, L, send_bag_size_required);    
    else
	for(int i = 0; i < 27; ++i)
	    sendpacks_start_host[i] = 0;
    
stage2:
    if (n > 0)
	PackingHalo::stage2 <<<(n + 127) / 128, 128>>>(p, n, L, send_bag, send_bag_size);

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaThreadSynchronize());

    if (send_bag_size < *send_bag_size_required_host)
    {
	printf("Ooops SIZE: %d REQUIRED: %d\n", send_bag_size, *send_bag_size_required_host);
	
	CUDA_CHECK(cudaFree(send_bag));
	CUDA_CHECK(cudaFree(acc_remote));
	send_bag_size = *send_bag_size_required_host;
	
	CUDA_CHECK(cudaMalloc(&send_bag, sizeof(Particle) * send_bag_size));
	CUDA_CHECK(cudaMalloc(&acc_remote, send_bag_size * sizeof(Acceleration)));
	
	goto stage2;
    }

    for(int i = 0; i < 27; ++i)
	send_offsets[i] = sendpacks_start_host[i];

    assert(send_offsets[26] == *send_bag_size_required_host);
    
    for(int i = 0; i < 26; ++i)
	MPI_CHECK( MPI_Isend(send_bag + send_offsets[i], send_offsets[i + 1] - send_offsets[i],
			     Particle::datatype(), dstranks[i], tagbase_dpd_remote_interactions + i, cartcomm, sendreq + i) );

    pending_send = true;
    
    {
	int sum = 0;
	
	for(int i = 0; i < 26; ++i)
	{
	    MPI_Status status;
	    MPI_CHECK( MPI_Probe(MPI_ANY_SOURCE, recv_tags[i], cartcomm, &status) );

	    int count;
	    MPI_CHECK( MPI_Get_count(&status, Particle::datatype(), &count) );

	    recv_offsets[i] = sum;
	    sum += count;
	}

	recv_offsets[26] = sum;
		
	if (recv_bag_size < sum)
	{
	    if (recv_bag_size > 0)
		CUDA_CHECK(cudaFree(recv_bag));
	    
	    CUDA_CHECK(cudaMalloc(&recv_bag, sizeof(Particle) * sum));
	    
	    recv_bag_size = sum;
	}
    }

    for(int i = 0; i < 26; ++i)
	MPI_CHECK( MPI_Irecv(recv_bag + recv_offsets[i], recv_offsets[i + 1] - recv_offsets[i],
			     Particle::datatype(), MPI_ANY_SOURCE, recv_tags[i], cartcomm, recvreq + i) );
}
    
void ComputeInteractionsDPD::dpd_remote_interactions_stage2(Particle * p, int n, int saru_tag1, Acceleration * a)
{
    MPI_Status statuses[26];
    MPI_CHECK( MPI_Waitall(26, recvreq, statuses) );
    
    int saru_tag2[26];
    bool saru_mask[26];
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = (coords[c] + d[c] + dims[c]) % dims[c];

	int indx[3];
	for(int c = 0; c < 3; ++c)
	    indx[c] = min(coords[c], coordsneighbor[c]) * dims[c] + max(coords[c], coordsneighbor[c]);

	saru_tag2[i] = indx[0] + dims[0] * dims[0] * (indx[1] + dims[1] * dims[1] * indx[2]);

	int dstrank;
	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, &dstrank) );

	if (dstrank != myrank)
	    saru_mask[i] = min(dstrank, myrank) == myrank;
	else
	{
	    int alter_ego = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	    saru_mask[i] = min(i, alter_ego) == i;
	}
    }

    const int nremote = send_offsets[26];

    if (nremote !=*send_bag_size_required_host)
	printf("about to abort: %d %d\n", nremote, *send_bag_size_required_host);
    
    assert(nremote ==*send_bag_size_required_host);
    assert(nremote <= send_bag_size);

    for(int i = 0; i < 26; ++i)
    {
	const int nd = send_offsets[i + 1] - send_offsets[i];
	const int ns = recv_offsets[i + 1] - recv_offsets[i];

	if (ns == 0 || nd == 0)
	    continue;

	PackingHalo::shift_remote_particles<<<(ns + 127) / 128, 128>>>(recv_bag + recv_offsets[i], ns, L, i);
	
#ifndef NDEBUG
	PackingHalo::check_send_particles <<<(nd + 127) / 128, 128 >>>(send_bag + send_offsets[i], nd, L, i);
#endif
	
	directforces_dpd_cuda_bipartite_nohost(
	    (float *)(send_bag + send_offsets[i]), (float *)(acc_remote + send_offsets[i]), nd,
	    (float *)(recv_bag + recv_offsets[i]), ns,
	    aij, gammadpd, sigma, 1. / sqrt(dt), saru_tag1, saru_tag2[i], saru_mask[i]);
    }

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaThreadSynchronize());

    PackingHalo::merge_accelerations<<<(nremote + 127) / 128, 128>>>(acc_remote, nremote, a, n, send_bag, p);
}

ComputeInteractionsDPD::ComputeInteractionsDPD(MPI_Comm cartcomm, int L):
    cartcomm(cartcomm), L(L), pending_send(false), recv_bag(NULL), recv_bag_size(0)
{
    assert(L % 2 == 0);
    assert(L >= 2);
	
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));
	
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
	
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = tagbase_dpd_remote_interactions + (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	    
	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];
	    
	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );
    }

    CUDA_CHECK(cudaHostAlloc((void **)&sendpacks_start, sizeof(int) * 27, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&sendpacks_start_host, sendpacks_start, 0));

    CUDA_CHECK(cudaHostAlloc((void **)&send_bag_size_required, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&send_bag_size_required_host, send_bag_size_required, 0));

    send_bag_size = L * L * 3 * 27;
    CUDA_CHECK(cudaMalloc(&send_bag, send_bag_size * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&acc_remote, send_bag_size * sizeof(Acceleration)));

    recv_bag_size = L * L * 3 * 27;
    CUDA_CHECK(cudaMalloc(&recv_bag, recv_bag_size * sizeof(Particle)));
	
    PackingHalo::setup<<<1, 1>>>(true);
}

ComputeInteractionsDPD::~ComputeInteractionsDPD()
{
    CUDA_CHECK(cudaFree(send_bag));
    CUDA_CHECK(cudaFree(acc_remote));
    
    CUDA_CHECK(cudaFree(recv_bag));

    CUDA_CHECK(cudaFreeHost(sendpacks_start));
    CUDA_CHECK(cudaFreeHost(send_bag_size_required));
}