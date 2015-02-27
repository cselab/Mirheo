#include <cassert>

#include <algorithm>

#include <cuda-dpd.h>

#include "dpd-interactions.h"

using namespace std;

ComputeInteractionsDPD::ComputeInteractionsDPD(MPI_Comm cartcomm): HaloExchanger(cartcomm, 0) {}

void ComputeInteractionsDPD::spawn_local_work()
{
    if (localwork.n > 0)
	forces_dpd_cuda_nohost((float *)localwork.p, (float *)localwork.a, localwork.n, 
			       localwork.cellsstart, localwork.cellscount,
			       1, LX, LY, LZ, aij, gammadpd, sigma, 1. / sqrt(dt), localwork.saru_tag);
}

void ComputeInteractionsDPD::evaluate(int& saru_tag, const Particle * const p, const int n, Acceleration * const a,
				      const int * const cellsstart, const int * const cellscount)
{
    localwork = LocalWorkParams(saru_tag, p, n, a, cellsstart, cellscount); 
    
    HaloExchanger::pack_and_post(p, n, cellsstart, cellscount); //spawn local work will be called within this function

    saru_tag += nranks - myrank;
    
    dpd_remote_interactions(p, n, saru_tag, a);
    
    saru_tag += 1 + myrank;  
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
	    atomicAdd(& alocal[pid].a[c], a.a[c]);
	    
	    assert(!isnan(a.a[c]));
	}
    }
}

void ComputeInteractionsDPD::dpd_remote_interactions(const Particle * const p, const int n, const int saru_tag1, Acceleration * const a)
{
    CUDA_CHECK(cudaPeekAtLastError());

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

	const int dstrank = dstranks[i];

	if (dstrank != myrank)
	    saru_mask[i] = min(dstrank, myrank) == myrank;
	else
	{
	    int alter_ego = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
	    saru_mask[i] = min(i, alter_ego) == i;
	}
    }

    CUDA_CHECK(cudaPeekAtLastError());

    //CUDA_CHECK(cudaStreamSynchronize(HaloExchanger::streams[0]));

    for(int i = 0; i < 26; ++i)
    {
	const int nd = sendhalos[i].dbuf.size;
	const int ns = recvhalos[i].dbuf.size;

	acc_remote[i].resize(nd);

	if (nd == 0)
	    continue;
	
#ifndef NDEBUG
	//fill acc entries with nan
	CUDA_CHECK(cudaMemset(acc_remote[i].data, 0xff, sizeof(Acceleration) * acc_remote[i].size));
#endif
	cudaStream_t mystream = streams[code2stream[i]];
	
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
					     ns, halosize[i], aij, gammadpd, sigma / sqrt(dt), saru_tag1, saru_tag2[i], saru_mask[i],
					     (float *)acc_remote[i].data);
	}
	else
	    directforces_dpd_cuda_bipartite_nohost(
		(float *)sendhalos[i].dbuf.data, (float *)acc_remote[i].data, nd,
		(float *)recvhalos[i].dbuf.data, ns,
		aij, gammadpd, sigma, 1. / sqrt(dt), saru_tag1, saru_tag2[i], saru_mask[i], mystream);
    }
    
    CUDA_CHECK(cudaPeekAtLastError());
    
    for(int i = 0; i < 26; ++i)
    {
	const int nd = acc_remote[i].size;
	
	if (nd > 0)
	    RemoteDPD::merge_accelerations<<<(nd + 127) / 128, 128, 0, streams[code2stream[i]]>>>
		(acc_remote[i].data, nd, a, n, sendhalos[i].dbuf.data, p, sendhalos[i].scattered_entries.data, myrank);
    }
   
    CUDA_CHECK(cudaPeekAtLastError());
}
