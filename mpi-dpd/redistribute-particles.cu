#include <cassert>
#include <vector>
#include <algorithm>

#include "redistribute-particles.h"

using namespace std;

RedistributeParticles::RedistributeParticles(MPI_Comm cartcomm, int L):
    cartcomm(cartcomm), L(L), pending_send(false), tmp(NULL), tmp_size(0)
{
    assert(L % 2 == 0);
	    
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
    }

    CUDA_CHECK(cudaHostAlloc((void **)&leaving_start_device, sizeof(int) * 28, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&leaving_start, leaving_start_device, 0));
}

namespace ParticleReordering
{
    __device__ int blockcount, global_histo[27];

    __global__ void setup()
    {
	blockcount = 0;

	for(int i = 0; i < 27; ++i)
	    global_histo[i] = 0;	
    }

    __global__ void count(Particle * p, int n, int L, int * leaving_start)
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
		assert(p[pid].x[c] >= -L/2 - L && p[pid].x[c] < L/2 + L);
	    
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

    __global__ void reorder(Particle * particles, int np, const int L, Particle * tmp)
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

    __global__ void shift(Particle * p, int np, int L, int code)
    {
	assert(blockDim.x * gridDim.x >= np);
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	
	int d[3] = { (code + 1) % 3 - 1, (code / 3 + 1) % 3 - 1, (code / 9 + 1) % 3 - 1 };

	if (pid >= np)
	    return;
	
	for(int c = 0; c < 3; ++c)
	    p[pid].x[c] -= d[c] * L;

#ifndef NDEBUG
	{
	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (p[pid].x[c] >= -L/2) + (p[pid].x[c] >= L/2)) % 3;
		
	    int newcode = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

	    if(newcode != 0)
		printf("ouch: new code is %d %d %d arriving from code %d -> %d %d %d\n", vcode[0], vcode[1], vcode[2], code,
		       d[0], d[1], d[2]);
	    
	    assert(newcode == 0);
	}
#endif
    }
}
    
int RedistributeParticles::stage1(Particle * p, int n)
{
    if (tmp_size < n)
    {
	if (tmp_size > 0)
	    CUDA_CHECK(cudaFree(tmp));

	CUDA_CHECK(cudaMalloc(&tmp, sizeof(Particle) * n));
	
	tmp_size = n;
    }
 
    ParticleReordering::setup<<<1, 1>>>();
    ParticleReordering::count<<<(n + 127) / 128, 128>>>(p, n, L, leaving_start_device);
    ParticleReordering::reorder<<<(n + 127) / 128, 128>>>(p, n, L, tmp);

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaThreadSynchronize());

    assert(leaving_start[27] == n);
   
    notleaving = leaving_start[1];
        
    MPI_Status statuses[26];
    if (pending_send)	    
	MPI_CHECK( MPI_Waitall(26, sendreq + 1, statuses) );

    for(int i = 1; i < 27; ++i)
	MPI_CHECK( MPI_Isend(tmp + leaving_start[i], leaving_start[i + 1] - leaving_start[i],
			     Particle::datatype(), rankneighbors[i], tagbase_redistribute_particles + i, cartcomm, sendreq + i) );
    
    pending_send = true;

    arriving = 0;
    arriving_start[0] = notleaving;
    for(int i = 1; i < 27; ++i)
    {
	MPI_Status status;
	MPI_CHECK( MPI_Probe(MPI_ANY_SOURCE, tagbase_redistribute_particles + i, cartcomm, &status) );
		
	int local_count;
	MPI_CHECK( MPI_Get_count(&status, Particle::datatype(), &local_count) );

	arriving_start[i] = notleaving + arriving;
	arriving += local_count;
    }
	    
    arriving_start[27] = notleaving + arriving;
    
    return notleaving + arriving;
}

void RedistributeParticles::stage2(Particle * p, int n)
{
    assert(n == notleaving + arriving);

    CUDA_CHECK(cudaMemcpy(p, tmp, sizeof(Particle) * notleaving, cudaMemcpyDeviceToDevice));
    
    for(int i = 1; i < 27; ++i)
	MPI_CHECK( MPI_Irecv(p + arriving_start[i], arriving_start[i + 1] - arriving_start[i], Particle::datatype(),
			     MPI_ANY_SOURCE, tagbase_redistribute_particles + i, cartcomm, recvreq + i) );

    MPI_Status statuses[26];	    
    MPI_CHECK( MPI_Waitall(26, recvreq + 1, statuses) );

    for(int i = 0; i < 27; ++i)
    {
	const int count = arriving_start[i + 1] - arriving_start[i];

	if (count == 0)
	    continue;

	ParticleReordering::shift<<<(count + 127) / 128, 128>>>(p + arriving_start[i], count, L, i);
    }
}

