#include "halo-exchanger.h"

#include "dpd-interactions.h"

using namespace std;

HaloExchanger::HaloExchanger(MPI_Comm cartcomm, int L):
    cartcomm(cartcomm), L(L), pending_send(false), recv_bag(NULL), send_bag(NULL), recv_bag_size(0), scattered_entries(NULL)
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
    CUDA_CHECK(cudaMalloc(&scattered_entries, send_bag_size * sizeof(int)));
    
    recv_bag_size = L * L * 3 * 27;
    CUDA_CHECK(cudaMalloc(&recv_bag, recv_bag_size * sizeof(Particle)));
}

namespace PackingHalo
{
    __device__ int blockcount, global_histo[27], requiredsize;

    __global__ void setup(bool firsttime)
    {
	blockcount = 0;

	for(int i = 0; i < 27; ++i)
	    global_histo[i] = 0;
    }

    template< int work >
    __global__ void count(int * const packs_start, const Particle * const p, const int np, const int L, int * bag_size_required)
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
	    }
	}
    }

    __global__ void pack(const Particle * const particles, int np, const int L, Particle * const bag, const int bagsize,
			 int * const scattered_entries)
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

    __global__ void shift_recv_particles(Particle * p, int n, int L, int code)
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
}

void HaloExchanger::pack_and_post(const Particle * const p, const int n)
{
    nlocal = n;
    
    MPI_Status statuses[26];
    
    if (pending_send)
	MPI_CHECK( MPI_Waitall(26, sendreq, statuses) );

    PackingHalo::setup<<<1, 1>>>(false);

    if (n > 0)
	PackingHalo::count<1> <<<(n + 127) / 128, 128>>> (sendpacks_start, p, n, L, send_bag_size_required);    
    else
	for(int i = 0; i < 27; ++i)
	    sendpacks_start_host[i] = 0;
    
stage2:
    if (n > 0)
	PackingHalo::pack <<<(n + 127) / 128, 128>>>(p, n, L, send_bag, send_bag_size, scattered_entries);

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaThreadSynchronize());

    if (send_bag_size < *send_bag_size_required_host)
    {
	printf("Ooops SIZE: %d REQUIRED: %d\n", send_bag_size, *send_bag_size_required_host);
	
	CUDA_CHECK(cudaFree(send_bag));
	CUDA_CHECK(cudaFree(scattered_entries));

	send_bag_size = *send_bag_size_required_host;
	
	CUDA_CHECK(cudaMalloc(&send_bag, sizeof(Particle) * send_bag_size));
	CUDA_CHECK(cudaMalloc(&scattered_entries, sizeof(int) * send_bag_size));
	
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
    
void HaloExchanger::wait_for_messages()
{
    MPI_Status statuses[26];
    MPI_CHECK( MPI_Waitall(26, recvreq, statuses) );
    
    const int nremote = send_offsets[26];

    if (nremote !=*send_bag_size_required_host)
	printf("about to abort: %d %d\n", nremote, *send_bag_size_required_host);
    
    assert(nremote ==*send_bag_size_required_host);
    assert(nremote <= send_bag_size);

    for(int i = 0; i < 26; ++i)
    {
	const int ns = recv_offsets[i + 1] - recv_offsets[i];

	if (ns > 0)
	    PackingHalo::shift_recv_particles<<<(ns + 127) / 128, 128>>>(recv_bag + recv_offsets[i], ns, L, i);
	
#ifndef NDEBUG
	const int nd = send_offsets[i + 1] - send_offsets[i];
	
	if (nd > 0)
	    PackingHalo::check_send_particles <<<(nd + 127) / 128, 128>>>(send_bag + send_offsets[i], nd, L, i);
#endif	
    }
}

int HaloExchanger::nof_sent_particles()
{
    const int nsend = send_offsets[26];

    if (nsend !=*send_bag_size_required_host)
	printf("about to abort: %d %d\n", nsend, *send_bag_size_required_host);
    
    assert(nsend ==*send_bag_size_required_host);
    assert(nsend <= send_bag_size);

    for(int i = 0; i < 26; ++i)
	assert(send_offsets[i + 1] - send_offsets[i] <= nlocal);

    return nsend;
}

HaloExchanger::~HaloExchanger()
{
    CUDA_CHECK(cudaFree(send_bag));
    CUDA_CHECK(cudaFree(recv_bag));
    CUDA_CHECK(cudaFreeHost(sendpacks_start));
    CUDA_CHECK(cudaFreeHost(send_bag_size_required));
}