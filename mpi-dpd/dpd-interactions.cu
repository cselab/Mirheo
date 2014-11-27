#include <cassert>

#include <algorithm>

#include <cuda-dpd.h>

#include "dpd-interactions.h"

using namespace std;

ComputeInteractionsDPD::ComputeInteractionsDPD(MPI_Comm cartcomm, int L):
    HaloExchanger(cartcomm, L)
{
    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaStreamCreate(streams + i));

    for(int i = 0, ctr = 1; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	const bool isface = abs(d[0]) + abs(d[1]) + abs(d[2]) == 1;

	code2stream[i] = 0;

	if (isface)
	{
	    code2stream[i] = ctr;
	    ctr++;
	}
    }
}

void ComputeInteractionsDPD::evaluate(int& saru_tag, const Particle * const p, const int n, Acceleration * const a,
				      const int * const cellsstart, const int * const cellscount)
{
    dpd_remote_interactions_stage1(p, n, cellsstart, cellscount);

    if (n > 0)
	forces_dpd_cuda_nohost((float *)p, (float *)a, n, 
			       cellsstart, cellscount,
			       1, L, L, L, aij, gammadpd, sigma, 1. / sqrt(dt), saru_tag);

    saru_tag += nranks - myrank;

    dpd_remote_interactions_stage2(p, n, saru_tag, a);

    saru_tag += 1 + myrank;  
}


__global__ void not_nan(float * p, const int n)
{
    assert(gridDim.x * blockDim.x >= n);

    const int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < n)
	assert(!isnan(p[gid]));
}

__global__ void fill_random(float * p, const int n)
{
    assert(gridDim.x * blockDim.x >= n);

    const int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid < n)
	p[gid] = 2 * (gid % 100) * 0.01 - 1;
}

namespace RemoteDPD
{
    __global__ void merge_accelerations(const Acceleration * const aremote, const int nremote,
					Acceleration * const alocal, const int nlocal,
					const Particle * premote, const Particle * plocal,
					const int * const scattered_entries, int rank)
    {
	assert(blockDim.x * gridDim.x >= nremote);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= nremote)
	    return;

	int pid = scattered_entries[ gid ];
	assert(pid >= 0 && pid < nlocal);

	Acceleration a = aremote[gid];

#ifndef NDEBUG
	Particle p1 = plocal[pid];
	Particle p2 = premote[gid];

	for(int c = 0; c < 3; ++c)
	{
	    assert(p1.x[c] == p2.x[c]);
	    assert(p1.x[c] == p2.x[c]);
	}

	for(int c = 0; c < 3; ++c)
	{
	    if (isnan(a.a[c]))
		printf("rank %d) oouch gid %d %f out of %d remote entries going to pid %d of %d particles\n", rank, gid, a.a[c], nremote, pid, nlocal);

	    assert(!isnan(a.a[c]));
	}
#endif
	for(int c = 0; c < 3; ++c)
	{
	    const float val = alocal[pid].a[c];
	    
	    alocal[pid].a[c] = val + a.a[c];
	    
	    assert(!isnan(val));
	}
    }
}

void ComputeInteractionsDPD::dpd_remote_interactions_stage1(const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount)
{
    HaloExchanger::pack_and_post(p, n, cellsstart, cellscount);
}

void ComputeInteractionsDPD::dpd_remote_interactions_stage2(const Particle * const p, const int n, const int saru_tag1, Acceleration * const a)
{
    wait_for_messages();

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

    for(int i = 0; i < 26; ++i)
    {
	const int nd = sendbufs[i].size;//send_counts[i];
	const int ns = recvbufs[i].size;//recv_counts[i];

	acc_remote[i].resize(nd);

	if (nd == 0)
	    continue;
	
#ifndef NDEBUG
	//fill acc entries with nan
	CUDA_CHECK(cudaMemset(acc_remote[i].data, 0xff, sizeof(Acceleration) * acc_remote[i].size));
#endif
	
	if (ns == 0)
	{
	    CUDA_CHECK(cudaMemset((float *)acc_remote[i].data, 0, nd * sizeof(Acceleration)));
	    continue;
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaStream_t mystream = streams[code2stream[i]];

	directforces_dpd_cuda_bipartite_nohost(
	    (float *)sendbufs[i].data, (float *)acc_remote[i].data, nd,
	    (float *)recvbufs[i].data, ns,
	    aij, gammadpd, sigma, 1. / sqrt(dt), saru_tag1, saru_tag2[i], saru_mask[i], mystream);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for(int i = 0; i < 26; ++i)
    {
	const int nd = acc_remote[i].size;
	
	if (nd > 0)
	    RemoteDPD::merge_accelerations<<<(nd + 127) / 128, 128>>>(acc_remote[i].data, nd, a, n,
								      sendbufs[i].data, p, scattered_entries[i].data, myrank);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
    }
   
    CUDA_CHECK(cudaPeekAtLastError());
}

ComputeInteractionsDPD::~ComputeInteractionsDPD()
{
    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaStreamDestroy(streams[i]));
}
