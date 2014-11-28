#include <cassert>
#include <vector>
#include <algorithm>

#include "redistribute-particles.h"

using namespace std;

static const int tagbase_redistribute_particles = 2115;

RedistributeParticles::RedistributeParticles(MPI_Comm _cartcomm, int L):
    L(L), pending_send(false), leaving_start(NULL), leaving_start_device(NULL)
{
    assert(L % 2 == 0);

    MPI_CHECK(MPI_Comm_dup(_cartcomm, &cartcomm));
	    
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
	    
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
	    
    for(int c = 0; c < 3; ++c)
	domain_extent[c] = L * dims[c];

    rankneighbors[0] = myrank;
    for(int i = 1; i < 27; ++i)
    {
	int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };
	
	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];
		
	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, rankneighbors + i) );

	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] - d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, anti_rankneighbors + i) );

	recvbufs[i].resize(L * L * 6);
	sendbufs[i].resize(L * L * 6);
    }

    CUDA_CHECK(cudaHostAlloc((void **)&leaving_start_device, sizeof(int) * 28, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&leaving_start, leaving_start_device, 0));

    CUDA_CHECK(cudaStreamCreate(&mystream));
}

//to keep the mess under a reasonable limit i wrap the set of
//kernels and data used for particle reordering within this namespace
namespace ParticleReordering
{
    __device__ int blockcount, global_histo[27];

    __global__ void setup()
    {
	blockcount = 0;

	for(int i = 0; i < 27; ++i)
	    global_histo[i] = 0;	
    }

    //this kernel is computing the histograms of particles landing into which subdomain
    //it creates a prefix sum to create global offsets (leaving_start) for the reordering (next kernel)
    __global__ void count(const Particle * p, const int n, const int L, int * const leaving_start)
    {
	assert(blockDim.x >= 27);
	assert(blockDim.x * gridDim.x >= n);

	const int tid = threadIdx.x;

	__shared__ int histo[27];
	
	if (tid < 27)
	    histo[tid] = 0;

	__syncthreads();

	const int pid = tid + blockDim.x * blockIdx.x;
	
	if (pid < n)
	{
	    for(int c = 0; c < 3; ++c)
	    {
		if (!(p[pid].x[c] >= -L/2 - L && p[pid].x[c] < L/2 + L))
		    printf("wow: pid %d component %d: %f\n", pid, c, p[pid].x[c]);

		assert(p[pid].x[c] >= -L/2 - L && p[pid].x[c] < L/2 + L);
	    }
	    
	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (p[pid].x[c] >= -L/2) + (p[pid].x[c] >= L/2)) % 3;
		
	    int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

	    atomicAdd(histo + code, 1);
	}

	__syncthreads();

	if (tid < 27 && histo[tid] > 0)
	    atomicAdd(global_histo + tid, histo[tid]);

	if (tid == 0)
	{
	    const int timestamp = atomicAdd(&blockcount, 1);

	    if (timestamp == gridDim.x - 1)
	    {
		blockcount = 0;

		int s = 0, curr;

		for(int i = 0; i < 27; ++i)
		{
		    curr = global_histo[i];
		    global_histo[i] = leaving_start[i] = s;
		    s += curr;
		}

		leaving_start[27] = s;
	    }
	}
    }

    __global__ void reorder(const Particle * const particles, const int np, const int L, Particle * const tmp)
    {
	assert(blockDim.x * gridDim.x >= np);
	assert(blockDim.x >= 27);
    
	__shared__ int histo[27];
	__shared__ int base[27];

	const int tid = threadIdx.x; 

	if (tid < 27)
	    histo[tid] = 0;

	__syncthreads();
	
	int offset, code;
	Particle p;
	
	const int pid = tid + blockDim.x * blockIdx.x;
	
	if (pid < np)
	{
	    p = particles[pid];
	    
	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (p.x[c] >= -L/2) + (p.x[c] >= L/2)) % 3;
		
	    code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
	    
	    offset = atomicAdd(histo + code, 1);
	}

	__syncthreads();
    
	if (tid < 27 && histo[tid] > 0)
	    base[tid] = atomicAdd(global_histo + tid, histo[tid]);

	__syncthreads();

	if (pid < np)
	{
	    if (!(base[code] + offset >= 0 && base[code] + offset < np))
	    {
		printf("ooops reordering::stage2: code %d base[code] %d offset %d np %d\n", code, base[code], offset, np);
	    }
	    assert(base[code] + offset >= 0 && base[code] + offset < np);
	}

	if (pid < np)
	    tmp[ base[code] + offset ] = p;
    }

#ifndef NDEBUG
    __global__ void check(Particle * const p, const int np, const int L, const int refcode, const int rank)
    {
	assert(blockDim.x * gridDim.x >= np);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (pid < np)
	{
	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (p[pid].x[c] >= -L/2) + (p[pid].x[c] >= L/2)) % 3;
	    
	    const int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

	    for(int c = 0; c < 3; ++c)
	    {
		if (vcode[c] == 0)
		    assert(p[pid].x[c] >= -L/2 && p[pid].x[c] < L/2);
		else if (vcode[c] == 1)
		    assert(p[pid].x[c] >= L/2 && p[pid].x[c] < L + L/2);
		else if (vcode[c] == 2)
		    assert(p[pid].x[c] <= -L/2 && p[pid].x[c] >= -L - L/2);
		else
		    asm("trap;");
	    }
	    assert(refcode == code);
	}
    }
#endif

    __global__ void shift(Particle * const p, const int np, const int L, const int code, const int rank)
    {
	assert(blockDim.x * gridDim.x >= np);
	
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	
	int d[3] = { (code + 1) % 3 - 1, (code / 3 + 1) % 3 - 1, (code / 9 + 1) % 3 - 1 };

	if (pid >= np)
	    return;

#ifndef NDEBUG
	Particle old = p[pid];
#endif
	Particle pnew = p[pid];

	for(int c = 0; c < 3; ++c)
	    pnew.x[c] -= d[c] * L;

	p[pid] = pnew;

#ifndef NDEBUG
	{
	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (pnew.x[c] >= -L/2) + (pnew.x[c] >= L/2)) % 3;
		
	    int newcode = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

	    if(newcode != 0)
		printf("rank %d) particle %d: ouch: new code is %d %d %d arriving from code %d -> %d %d %d \np: %f %f %f (before: %f %f %f)\n", 
		       rank,  pid, vcode[0], vcode[1], vcode[2], code,
		       d[0], d[1], d[2], pnew.x[0], pnew.x[1], pnew.x[2],
		       old.x[0], old.x[1], old.x[2]);
	    
	    assert(newcode == 0);
	}
#endif
    }
}

int RedistributeParticles::stage1(const Particle * const p, const int n)
{  
    MPI_Status statuses[26];

    if (pending_send)	    
	MPI_CHECK( MPI_Waitall(26, sendreq, statuses) );
    
    reordered.resize(n);
    
    ParticleReordering::setup<<<1, 1, 0, mystream>>>();
    
    if (n > 0)
	ParticleReordering::count<<< (n + 127) / 128, 128, 0, mystream >>>(p, n, L, leaving_start_device);
    else
	for(int i = 0; i < 28; ++i)
	    leaving_start[i] = 0;
    
    if (n > 0)
	ParticleReordering::reorder<<< (n + 127) / 128, 128, 0, mystream >>>(p, n, L, reordered.data);
    
    CUDA_CHECK(cudaPeekAtLastError());
   

#ifndef NDEBUG    

    CUDA_CHECK(cudaStreamSynchronize(mystream));

    assert(leaving_start[0] == 0);
    assert(leaving_start[27] == n); 
   
    for(int i = 0; i < 27; ++i)
    {
	const int count = leaving_start[i + 1] - leaving_start[i];

	if (count > 0)
	    ParticleReordering::check<<< (count + 127) / 128, 128, 0, mystream >>>(reordered.data + leaving_start[i], count, L, i, myrank);

	CUDA_CHECK(cudaPeekAtLastError());
    }
#endif

    CUDA_CHECK(cudaStreamSynchronize(mystream));

    notleaving = leaving_start[1];

    {
	MPI_Request sendcountreq[26];

	send_counts[0] = 0;
	
	for(int i = 1; i < 27; ++i)
	{
	    const int count = (leaving_start[i + 1] - leaving_start[i]);
	    
	    send_counts[i] = count;
	    
	    MPI_CHECK( MPI_Isend(send_counts + i, 1, MPI_INTEGER, rankneighbors[i],  
				 tagbase_redistribute_particles + i + 377, cartcomm, 
				 &sendcountreq[i-1]) );
	}
	
	recv_counts[0] = notleaving;

	arriving = 0;

	arriving_start[0] = 0;

	MPI_Status status;
	for(int i = 1; i < 27; ++i)
	{
	    MPI_CHECK( MPI_Recv(recv_counts + i, 1, MPI_INTEGER, anti_rankneighbors[i], 
				tagbase_redistribute_particles + i + 377, cartcomm, &status) );

	    arriving_start[i] = notleaving + arriving;

	    arriving += recv_counts[i];
	}

	arriving_start[27] = notleaving + arriving;
	
	MPI_Status statuses[26];	    
	MPI_CHECK( MPI_Waitall(26, sendcountreq, statuses) );
    }
     
    //cuda-aware mpi receive
    for(int i = 1; i < 27; ++i)
    {
	const int count = recv_counts[i];
	    
	recvbufs[i].resize(count);
	
	MPI_CHECK( MPI_Irecv(recvbufs[i].data, count * 6, /*Particle::datatype()*/MPI_FLOAT,
			     anti_rankneighbors[i], tagbase_redistribute_particles + i, cartcomm, 
			     &recvreq[i-1]) );
    }

    for(int i = 1; i < 27; ++i)
    {
	const int count = send_counts[i];

	sendbufs[i].resize(count);
	
	CUDA_CHECK(cudaMemcpyAsync(sendbufs[i].data, reordered.data + leaving_start[i], 
				   sizeof(Particle) * count, cudaMemcpyDeviceToDevice, mystream));
    }

    CUDA_CHECK(cudaStreamSynchronize(mystream));
    
    for(int i = 1; i < 27; ++i)
    {
	const int count = send_counts[i];

	MPI_CHECK( MPI_Isend(sendbufs[i].data, count * 6,
			     MPI_FLOAT, rankneighbors[i], tagbase_redistribute_particles + i, 
			     cartcomm, &sendreq[i-1]) );
    }

    CUDA_CHECK(cudaPeekAtLastError());

    pending_send = true;

    return notleaving + arriving;
}

void RedistributeParticles::stage2(Particle * const p, const int n)
{
    assert(n == notleaving + arriving);

    CUDA_CHECK(cudaMemcpy(p, reordered.data, sizeof(Particle) * notleaving, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaPeekAtLastError());
    
    MPI_Status statuses[26];	    
    MPI_CHECK( MPI_Waitall(26, recvreq, statuses) );
    
    for(int i = 1; i < 27; ++i)
    { 
	const int count = recv_counts[i];

	if (count == 0)
	    continue;

	CUDA_CHECK(cudaMemcpyAsync(p + arriving_start[i], recvbufs[i].data, 
				   sizeof(Particle) * count, cudaMemcpyDeviceToDevice, mystream));
	
	ParticleReordering::shift<<< (count + 127) / 128, 128, 0, mystream >>>(p + arriving_start[i], count, L, i, myrank);
    }

#ifndef NDEBUG
    if (n > 0)
	ParticleReordering::check<<< (n + 127) / 128, 128, 0, mystream >>>(p, n, L, 0, myrank);
#endif

    CUDA_CHECK(cudaPeekAtLastError());
}

RedistributeParticles::~RedistributeParticles()
{
    CUDA_CHECK(cudaStreamDestroy(mystream));
    CUDA_CHECK(cudaFreeHost(leaving_start_device));

    MPI_CHECK(MPI_Comm_free(&cartcomm));
}
