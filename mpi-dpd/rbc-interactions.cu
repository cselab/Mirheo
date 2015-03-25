/*
 *  rbc-interactions.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <../dpd-rng.h>

#include "rbc-interactions.h"
#include "minmax-massimo.h"

namespace KernelsRBC
{
    struct ParamsFSI
    {
	float aij, gamma, sigmaf;
    };

    __constant__ ParamsFSI params;
    
    texture<float2, cudaTextureType1D> texSolventParticles;
    texture<int, cudaTextureType1D> texCellsStart, texCellsCount;

    static bool firsttime = true;
    
    __global__ void fsi_forces(const float seed,
			       Acceleration * accsolvent, const int npsolvent,
			       const Particle * const particle, const int nparticles, Acceleration * accrbc);
    
    void setup(const Particle * const solvent, const int npsolvent, const int * const cellsstart, const int * const cellscount)
    {
	if (firsttime)
	{
	    texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
	    texCellsStart.filterMode = cudaFilterModePoint;
	    texCellsStart.mipmapFilterMode = cudaFilterModePoint;
	    texCellsStart.normalized = 0;
    
	    texCellsCount.channelDesc = cudaCreateChannelDesc<int>();
	    texCellsCount.filterMode = cudaFilterModePoint;
	    texCellsCount.mipmapFilterMode = cudaFilterModePoint;
	    texCellsCount.normalized = 0;

	    texSolventParticles.channelDesc = cudaCreateChannelDesc<float2>();
	    texSolventParticles.filterMode = cudaFilterModePoint;
	    texSolventParticles.mipmapFilterMode = cudaFilterModePoint;
	    texSolventParticles.normalized = 0;
	    firsttime = false;
	}
	
	size_t textureoffset;
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texSolventParticles, solvent, &texSolventParticles.channelDesc,
				   sizeof(float) * 6 * npsolvent));

	const int ncells = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;
	
	assert(textureoffset == 0);
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart, &texCellsStart.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0); 
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsCount, cellscount, &texCellsCount.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);

	CUDA_CHECK(cudaFuncSetCacheConfig(fsi_forces, cudaFuncCachePreferL1));
    }
    
    __global__ void shift_send_particles(const Particle * const src, const int n, const int code, Particle * const dst)
    {
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	if (gid < n)
	{
	    Particle p = src[gid];
	    
	    for(int c = 0; c < 3; ++c)
		p.x[c] -= d[c] * L[c];

	    dst[gid] = p;
	}
    }

    __device__ bool fsi_kernel(const float seed,
			       const int dpid, const float3 xp, const float3 up, const int spid,
			       float& xforce, float& yforce, float& zforce)
    {
	xforce = yforce = zforce = 0;
	
	const int sentry = 3 * spid;
	
	const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry);
	const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
	const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);
	
	const float _xr = xp.x - stmp0.x;
	const float _yr = xp.y - stmp0.y;
	const float _zr = xp.z - stmp1.x;

	const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	
	if (rij2 > 1)
	    return false;
	
	const float invrij = rsqrtf(rij2);
	
	const float rij = rij2 * invrij;
	const float argwr = max((float)0, 1 - rij);
	const float wr = powf(argwr, powf(0.5f, -VISCOSITY_S_LEVEL));
	
	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;
	
	const float rdotv = 
	    xr * (up.x - stmp1.y) +
	    yr * (up.y - stmp2.x) +
	    zr * (up.z - stmp2.y);
	
	//const float mysaru = saru(saru_tag, dpid, spid);
	//const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
	const float myrandnr = Logistic::mean0var1(seed, dpid, spid);
	
	const float strength = params.aij * argwr +  (- params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;
	
	xforce = strength * xr;
	yforce = strength * yr;
	zforce = strength * zr; 

	return true;
    }

    
    __global__ void fsi_forces(const float seed,
			       Acceleration * accsolvent, const int npsolvent,
			       const Particle * const particle, const int nparticles, Acceleration * accrbc)
    {
	const int dpid = threadIdx.x + blockDim.x * blockIdx.x;

	if (dpid >= nparticles)
	    return;

	const Particle p = particle[dpid];

	const float3 xp = make_float3(p.x[0], p.x[1], p.x[2]);
	const float3 up = make_float3(p.u[0], p.u[1], p.u[2]);
		
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	int mycid[3];
	for(int c = 0; c < 3; ++c)
	    mycid[c] = (int)floor(p.x[c] + L[c]/2);

	float fsum[3] = {0, 0, 0};
	
	for(int code = 0; code < 27; ++code)
	{
	    const int d[3] = {
		(code % 3) - 1,
		(code/3 % 3) - 1,
		(code/9 % 3) - 1
	    };
	    
	    int vcid[3];
	    for(int c = 0; c < 3; ++c)
		vcid[c] = mycid[c] + d[c];

	    bool validcid = true;
	    for(int c = 0; c < 3; ++c)
		validcid &= vcid[c] >= 0 && vcid[c] < L[c];

	    if (!validcid)
		continue;
	    
	    const int cid = vcid[0] + XSIZE_SUBDOMAIN * (vcid[1] + YSIZE_SUBDOMAIN * vcid[2]);
	    const int mystart = tex1Dfetch(texCellsStart, cid);
	    const int myend = mystart + tex1Dfetch(texCellsCount, cid);
	    
	    assert(mystart >= 0 && mystart <= myend);
	    assert(myend <= npsolvent);

	    #pragma unroll 4
	    for(int s = mystart; s < myend; ++s)
	    {
		float f[3];
		const bool nonzero = fsi_kernel(seed, dpid, xp, up, s, f[0], f[1], f[2]);

		if (nonzero)
		{
		    for(int c = 0; c < 3; ++c)
			fsum[c] += f[c];
		     
		    for(int c = 0; c < 3; ++c)
		    	   atomicAdd(c + (float *)(accsolvent + s), -f[c]);
		}
	    }
	}
	
	for(int c = 0; c < 3; ++c)
	    accrbc[dpid].a[c] = fsum[c];
    }

    __global__ void merge_accelerations(const Acceleration * const src, const int n, Acceleration * const dst)
    {	
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid < n)
	    for(int c = 0; c < 3; ++c)
		dst[gid].a[c] += src[gid].a[c];
    }
}

ComputeInteractionsRBC::ComputeInteractionsRBC(MPI_Comm _cartcomm): nvertices(0)
{ 
    assert(XSIZE_SUBDOMAIN % 2 == 0 && YSIZE_SUBDOMAIN % 2 == 0 && ZSIZE_SUBDOMAIN % 2 == 0);
    assert(XSIZE_SUBDOMAIN >= 2 && YSIZE_SUBDOMAIN >= 2 && ZSIZE_SUBDOMAIN >= 2);
    
    CudaRBC::Extent host_extent;
    CudaRBC::setup(nvertices, host_extent);
    
    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
 
    local_trunk = Logistic::KISS(1908 - myrank, 1409 + myrank, 290, 12968);

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
    }

    KernelsRBC::ParamsFSI params = {aij, gammadpd, sigmaf};
    
    CUDA_CHECK(cudaMemcpyToSymbol(KernelsRBC::params, &params, sizeof(KernelsRBC::ParamsFSI)));
    
    CUDA_CHECK(cudaEventCreate(&evextents, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&evfsi, cudaEventDisableTiming));
}

void ComputeInteractionsRBC::_compute_extents(const Particle * const rbcs, const int nrbcs, cudaStream_t stream)
{
#if 1
    minmax_massimo(rbcs, nvertices, nrbcs, minextents.devptr, maxextents.devptr, stream);
#else
    for(int i = 0; i < nrbcs; ++i)
	CudaRBC::extent_nohost(stream, (float *)(rbcs + nvertices * i), extents.devptr + i);
#endif
}

void ComputeInteractionsRBC::pack_and_post(const Particle * const rbcs, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RBC/pack-post", NVTX_C2);

    minextents.resize(nrbcs);
    maxextents.resize(nrbcs);

    _compute_extents(rbcs, nrbcs, stream);

    CUDA_CHECK(cudaEventRecord(evextents));
    CUDA_CHECK(cudaEventSynchronize(evextents));

    for(int i = 0; i < 26; ++i)
	haloreplica[i].clear();

    for(int i = 0; i < nrbcs; ++i)
    {
	float pmin[3] = {minextents.data[i].x, minextents.data[i].y, minextents.data[i].z};
	float pmax[3] = {maxextents.data[i].x, maxextents.data[i].y, maxextents.data[i].z};

	for(int code = 0; code < 26; ++code)
	{
	    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	    const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	    bool interacting = true;
	    
	    for(int c = 0; c < 3; ++c)
	    {
		const float range_start = max((float)(d[c] * L[c] - L[c]/2 - 1), pmin[c]);
		const float range_end = min((float)(d[c] * L[c] + L[c]/2 + 1), pmax[c]);

		interacting &= range_end > range_start;
	    }

	    if (interacting)
		haloreplica[code].push_back(i);
	}
    }

    MPI_Request reqrecvcounts[26];
    for(int i = 0; i <26; ++i)
	MPI_CHECK(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i], recv_tags[i] + 2077, cartcomm, reqrecvcounts + i));

    MPI_Request reqsendcounts[26];
    for(int i = 0; i < 26; ++i)
    {
	send_counts[i] = haloreplica[i].size();
	MPI_CHECK(MPI_Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i], i + 2077, cartcomm, reqsendcounts + i));
    }

    {
	MPI_Status statuses[26];
	MPI_CHECK(MPI_Waitall(26, reqrecvcounts, statuses));
	MPI_CHECK(MPI_Waitall(26, reqsendcounts, statuses));
    }

    for(int i = 0; i < 26; ++i)
	local[i].setup(send_counts[i] * nvertices);

    for(int i = 0; i < 26; ++i)
    {
	for(int j = 0; j < haloreplica[i].size(); ++j)
	    KernelsRBC::shift_send_particles<<< (nvertices + 127) / 128, 128, 0, stream>>>
		(rbcs + nvertices * haloreplica[i][j], nvertices, i, local[i].state.devptr + nvertices * j);
	 
	CUDA_CHECK(cudaPeekAtLastError());
    }
     
    CUDA_CHECK(cudaEventRecord(evfsi));
    CUDA_CHECK(cudaEventSynchronize(evfsi));

    for(int i = 0; i < 26; ++i)
	remote[i].setup(recv_counts[i] * nvertices);

    for(int i = 0; i < 26; ++i)
	if (recv_counts[i] > 0)
	{
	    MPI_Request request;
	    
	    MPI_CHECK(MPI_Irecv(remote[i].state.data, recv_counts[i] * nvertices, Particle::datatype(), dstranks[i],
				recv_tags[i] + 2011, cartcomm, &request));

	    reqrecvp.push_back(request);
	}

    for(int i = 0; i < 26; ++i)
	if (send_counts[i] > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(local[i].result.data, send_counts[i] * nvertices, Acceleration::datatype(), dstranks[i],
				recv_tags[i] + 2285, cartcomm, &request));

	    reqrecvacc.push_back(request);
	    
	    MPI_CHECK(MPI_Isend(local[i].state.data, send_counts[i] * nvertices, Particle::datatype(), dstranks[i],
				i + 2011, cartcomm, &request));

	    reqsendp.push_back(request);
	}
}

void ComputeInteractionsRBC::_internal_forces(const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
{
    CudaRBC::forces_nohost(stream, nrbcs, (float *)rbcs, (float *)accrbc);
}

void ComputeInteractionsRBC::evaluate(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
				      const int * const cellsstart_solvent, const int * const cellscount_solvent,
				      const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream)
{	
    KernelsRBC::setup(solvent, nparticles, cellsstart_solvent, cellscount_solvent);

    pack_and_post(rbcs, nrbcs, stream);

    if (nrbcs > 0 && nparticles > 0)
    {
	NVTX_RANGE("RBC/local forces", NVTX_C3);

	KernelsRBC::fsi_forces<<< (nrbcs * nvertices + 127) / 128, 128, 0, stream >>>
	    (local_trunk.get_float(), accsolvent, nparticles, rbcs, nrbcs * nvertices, accrbc);
		
	_internal_forces(rbcs, nrbcs, accrbc, stream);
    }
    
    {
	NVTX_RANGE("RBC/wait-exchange", NVTX_C4);
	_wait(reqrecvp);
	_wait(reqsendp);
    }
    
    {
	NVTX_RANGE("RBC/fsi", NVTX_C5);
	for(int i = 0; i < 26; ++i)
	{
	    const int count = remote[i].state.size;

	    if (count > 0)
		KernelsRBC::fsi_forces<<< (count + 127) / 128, 128, 0, stream >>>
		    (local_trunk.get_float(), accsolvent, nparticles, remote[i].state.devptr, count, remote[i].result.devptr);
	}
	
	CUDA_CHECK(cudaEventRecord(evfsi));
	CUDA_CHECK(cudaEventSynchronize(evfsi));
    }

    {
	NVTX_RANGE("RBC/send-results", NVTX_C6);

	for(int i = 0; i < 26; ++i)
	    if (recv_counts[i] > 0)
	    {
		MPI_Request request;
		
		MPI_CHECK(MPI_Isend(remote[i].result.data, recv_counts[i] * nvertices, Acceleration::datatype(), dstranks[i],
				i + 2285, cartcomm, &request));
		
		reqsendacc.push_back(request);
	    }

	_wait(reqrecvacc);
	_wait(reqsendacc);
    }
    
    {
	NVTX_RANGE("RBC/merge", NVTX_C7);

	for(int i = 0; i < 26; ++i)
	    for(int j = 0; j < haloreplica[i].size(); ++j)
		KernelsRBC::merge_accelerations<<< (nvertices + 127) / 128, 128, 0, stream>>>(local[i].result.devptr + nvertices * j, nvertices,
											      accrbc + nvertices * haloreplica[i][j]);
    }
}

ComputeInteractionsRBC::~ComputeInteractionsRBC()
{
    MPI_CHECK(MPI_Comm_free(&cartcomm));

    CUDA_CHECK(cudaEventDestroy(evextents));
    CUDA_CHECK(cudaEventDestroy(evfsi));
}

