#include <cassert>

#include <algorithm>

#include <cuda-dpd.h>

#include "dpd-interactions.h"

using namespace std;

ComputeInteractionsDPD::ComputeInteractionsDPD(MPI_Comm cartcomm): 
    HaloExchanger(cartcomm, 0), local_trunk(0, 0, 0, 0)
{
    int myrank;
    MPI_CHECK(MPI_Comm_rank(cartcomm, &myrank));
    
    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = (coords[c] + d[c] + dims[c]) % dims[c];

	int indx[3];
	for(int c = 0; c < 3; ++c)
	    indx[c] = min(coords[c], coordsneighbor[c]) * dims[c] + max(coords[c], coordsneighbor[c]);

	const int interrank_seed_base = indx[0] + dims[0] * dims[0] * (indx[1] + dims[1] * dims[1] * indx[2]);
	
	int interrank_seed_offset;

	{
	    const bool isplus = 
		d[0] + d[1] + d[2] > 0 || 
		d[0] + d[1] + d[2] == 0 && (
		    d[0] > 0 || d[0] == 0 && (
			d[1] > 0 || d[1] == 0 && d[2] > 0
			)
		    );
	    
	    const int mysign = 2 * isplus - 1;

	    int v[3] = { 1 + mysign * d[0], 1 + mysign * d[1], 1 + mysign *d[2] };
	    
	    interrank_seed_offset = v[0] + 3 * (v[1] + 3 * v[2]);
	    }

	const int interrank_seed = interrank_seed_base + interrank_seed_offset;
	
	interrank_trunks[i] = Logistic::KISS(390 + interrank_seed, interrank_seed  + 615, 12309, 23094); 

	const int dstrank = dstranks[i];

	if (dstrank != myrank)
	    interrank_masks[i] = min(dstrank, myrank) == myrank;
	else
	{
	    int alter_ego = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	    interrank_masks[i] = min(i, alter_ego) == i;
	}
    }
    
    CUDA_CHECK(cudaEventCreate(&evmerge, cudaEventDisableTiming));
}


void ComputeInteractionsDPD::spawn_local_work()
{
    NVTX_RANGE("DPD/bulk", NVTX_C5)
	
    if (localwork.n > 0)
	forces_dpd_cuda_nohost((float *)localwork.p, (float *)localwork.a, localwork.n, 
			       localwork.cellsstart, localwork.cellscount,
			       1, XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN, aij, gammadpd, 
			       sigma, 1. / sqrt(dt), localwork.seed1, localwork.stream);
}

void ComputeInteractionsDPD::evaluate(const Particle * const p, const int n, Acceleration * const a,
				      const int * const cellsstart, const int * const cellscount, cudaStream_t stream)
{
    localwork = LocalWorkParams(local_trunk.get_float(), p, n, a, cellsstart, cellscount, stream); 
    
    HaloExchanger::pack_and_post(p, n, cellsstart, cellscount); //spawn local work will be called within this function
    
    dpd_remote_interactions(p, n, a);
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
		printf("rank %d) oouch gid %d %f out of %d remote entries going to pid %d of %d particles\n", 
		       rank, gid, a.a[c], nremote, pid, nlocal);

	    assert(!isnan(a.a[c]));
	}
#endif
	for(int c = 0; c < 3; ++c)
	{
	    atomicAdd(& alocal[pid].a[c], a.a[c]);
	    
	    assert(!isnan(a.a[c]));
	}
    }
}

void ComputeInteractionsDPD::dpd_remote_interactions(const Particle * const p, const int n, Acceleration * const a)
{
    CUDA_CHECK(cudaPeekAtLastError());
    
    wait_for_messages();
    
    CUDA_CHECK(cudaPeekAtLastError());
        
    NVTX_RANGE("DPD/bipartite", NVTX_C3)
	
    for(int i = 0; i < 26; ++i)
    {
	const float interrank_seed = interrank_trunks[i].get_float();
	

	const int nd = sendhalos[i].dbuf.size;
	const int ns = recvhalos[i].dbuf.size;

	acc_remote[i].resize(nd);

	if (nd == 0)
	    continue;

	cudaStream_t mystream = streams[code2stream[i]];
	
#ifndef NDEBUG
	//fill acc entries with nan
	CUDA_CHECK(cudaMemsetAsync(acc_remote[i].data, 0xff, sizeof(Acceleration) * acc_remote[i].size, mystream));
#endif
	if (ns == 0)
	{
	    CUDA_CHECK(cudaMemsetAsync((float *)acc_remote[i].data, 0, nd * sizeof(Acceleration), mystream));
	    continue;
	}
	
	if (sendhalos[i].dcellstarts.size * recvhalos[i].dcellstarts.size > 1 && nd * ns > 10 * 10)
	{	   

	    texDC[i].acquire(sendhalos[i].dcellstarts.data, sendhalos[i].dcellstarts.capacity);
	    texSC[i].acquire(recvhalos[i].dcellstarts.data, recvhalos[i].dcellstarts.capacity);
	    texSP[i].acquire((float2*)recvhalos[i].dbuf.data, recvhalos[i].dbuf.capacity * 3);
	    
	    forces_dpd_cuda_bipartite_nohost(mystream, (float2 *)sendhalos[i].dbuf.data, nd, texDC[i].texObj, texSC[i].texObj, texSP[i].texObj,
					     ns, halosize[i], aij, gammadpd, sigma / sqrt(dt), interrank_seed, interrank_masks[i],
					     (float *)acc_remote[i].data);
	}
	else
	    directforces_dpd_cuda_bipartite_nohost(
		(float *)sendhalos[i].dbuf.data, (float *)acc_remote[i].data, nd,
		(float *)recvhalos[i].dbuf.data, ns,
		aij, gammadpd, sigma, 1. / sqrt(dt), interrank_seed, interrank_masks[i], mystream);
	    
    }
    
    CUDA_CHECK(cudaPeekAtLastError());
    
    for(int i = 0; i < 26; ++i)
    {
	const int nd = acc_remote[i].size;
	
	if (nd > 0)
	    RemoteDPD::merge_accelerations<<<(nd + 127) / 128, 128, 0, streams[code2stream[i]]>>>
		(acc_remote[i].data, nd, a, n, sendhalos[i].dbuf.data, p, sendhalos[i].scattered_entries.data, myrank);
    }
    CUDA_CHECK(cudaEventRecord(evmerge));
    CUDA_CHECK(cudaEventSynchronize(evmerge));
   
    CUDA_CHECK(cudaPeekAtLastError());
}

ComputeInteractionsDPD::~ComputeInteractionsDPD()
{
    CUDA_CHECK(cudaEventDestroy(evmerge));
}