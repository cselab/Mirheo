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
    
    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaEventCreate(evremoteint + i, cudaEventDisableTiming));

    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaStreamCreate(streams + i));

    for(int i = 0, ctr = 1; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	const bool isface = abs(d[0]) + abs(d[1]) + abs(d[2]) == 1;

	code2stream[i] = i % 7;

	if (isface)
	{
	    code2stream[i] = ctr;
	    ctr++;
	}
    }
}

void ComputeInteractionsDPD::local_interactions(const Particle * const p, const int n, Acceleration * const a,
						const int * const cellsstart, const int * const cellscount, cudaStream_t stream)
{
    NVTX_RANGE("DPD/local", NVTX_C5);
    
    if (n > 0)
	forces_dpd_cuda_nohost((float *)p, (float *)a, n, 
			       cellsstart, cellscount,
			       1, XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN, aij, gammadpd, 
			       sigma, 1. / sqrt(dt), local_trunk.get_float(), stream);
}

namespace RemoteDPD
{
    int npackedparticles;
    
    __constant__ int packstarts[27];
    __constant__ int * scattered_indices[26];
    __constant__ Acceleration * remote_accelerations[26];
    
    __global__ void merge_all(Acceleration * const alocal, const int nlocal, const int nremote)
    {
	assert(blockDim.x * gridDim.x >= nremote);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= packstarts[26])
	    return;

	const int key9 = 9 * ((gid >= packstarts[9]) + (gid >= packstarts[18]));
	const int key3 = 3 * ((gid >= packstarts[key9 + 3]) + (gid >= packstarts[key9 + 6]));
	const int key1 = (gid >= packstarts[key9 + key3 + 1]) + (gid >= packstarts[key9 + key3 + 2]);
	const int idpack = key9 + key3 + key1;

	assert(idpack >= 0 && idpack < 26);
	assert(gid >= packstarts[idpack] && gid < packstarts[idpack + 1]);

	const int offset = gid - packstarts[idpack];
	
	int pid = scattered_indices[idpack][offset];

	if (!(pid >= 0 && pid < nlocal))
	    printf("oooooops pid is %d whereas nlocal is %d\n", pid, nlocal);
	assert(pid >= 0 && pid < nlocal);

	Acceleration a = remote_accelerations[idpack][offset];

	for(int c = 0; c < 3; ++c)
	    assert(!isnan(a.a[c]));
	
	for(int c = 0; c < 3; ++c)
	    atomicAdd(& alocal[pid].a[c], a.a[c]);
    }
    
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

void ComputeInteractionsDPD::remote_interactions(const Particle * const p, const int n, Acceleration * const a, cudaStream_t stream)
{
    CUDA_CHECK(cudaPeekAtLastError());

    {
	NVTX_RANGE("DPD/remote", NVTX_C3);

	for(int i = 0; i < 7; ++i)
	    CUDA_CHECK(cudaStreamWaitEvent(streams[i], evshiftrecvp, 0));

	for(int pass = 0; pass < 2; ++pass)
	{
	    const bool face_pass = pass == 0;
	    
	    for(int i = 0; i < 26; ++i)
	    {
		int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };
	    
		const bool isface = abs(d[0]) + abs(d[1]) + abs(d[2]) == 1;
	    
		if (isface != face_pass)
		    continue;
		
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
	}

	for(int i = 0; i < 7; ++i)
	    CUDA_CHECK(cudaEventRecord(evremoteint[i]));
        
	CUDA_CHECK(cudaPeekAtLastError());
    }

    {
	NVTX_RANGE("DPD/merge", NVTX_C6);

	{
	    int packstarts[27];
	    
	    packstarts[0] = 0;
	    for(int i = 0, s = 0; i < 26; ++i)
		packstarts[i + 1] =  (s += acc_remote[i].size * (sendhalos[i].expected > 0));
	    	    
	    RemoteDPD::npackedparticles = packstarts[26];
	    
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(RemoteDPD::packstarts, packstarts,
					       sizeof(packstarts), 0, cudaMemcpyHostToDevice, stream));
	}

	{
	    int * scattered_indices[26];
	    for(int i = 0; i < 26; ++i)
		scattered_indices[i] = sendhalos[i].scattered_entries.data;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(RemoteDPD::scattered_indices, scattered_indices,
					       sizeof(scattered_indices), 0, cudaMemcpyHostToDevice, stream));
	}
	
	{
	    Acceleration * remote_accelerations[26];

	    for(int i = 0; i < 26; ++i)
		remote_accelerations[i] = acc_remote[i].data;

	    CUDA_CHECK(cudaMemcpyToSymbolAsync(RemoteDPD::remote_accelerations, remote_accelerations,
					       sizeof(remote_accelerations), 0, cudaMemcpyHostToDevice, stream));
	}
	
	
	for(int i = 0; i < 7; ++i)
	    CUDA_CHECK(cudaStreamWaitEvent(stream, evremoteint[i], 0));

#if 1
	RemoteDPD::merge_all<<< (RemoteDPD::npackedparticles + 127) / 128, 128, 0, stream >>>(a, n, RemoteDPD::npackedparticles);
#else
	for(int i = 0; i < 26; ++i)
	{
	    const int nd = acc_remote[i].size;
	    
	    if (nd > 0)
		RemoteDPD::merge_accelerations<<<(nd + 127) / 128, 128, 0, streams[code2stream[i]]>>>
		    (acc_remote[i].data, nd, a, n, sendhalos[i].dbuf.data, p, sendhalos[i].scattered_entries.data, myrank);
	}
#endif
	CUDA_CHECK(cudaPeekAtLastError());
    }
}

ComputeInteractionsDPD::~ComputeInteractionsDPD()
{
    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaStreamDestroy(streams[i]));
    
    for(int i = 0; i < 7; ++i)
	CUDA_CHECK(cudaEventDestroy(evremoteint[i]));
}