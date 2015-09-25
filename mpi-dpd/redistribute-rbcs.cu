/*
 *  redistribute-rbcs.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-01.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <vector>

#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "minmax.h"

RedistributeRBCs::RedistributeRBCs(MPI_Comm _cartcomm): nvertices(CudaRBC::get_nvertices())
{
    assert(XSIZE_SUBDOMAIN % 2 == 0 && YSIZE_SUBDOMAIN % 2 == 0 && ZSIZE_SUBDOMAIN % 2 == 0);
    assert(XSIZE_SUBDOMAIN >= 2 && YSIZE_SUBDOMAIN >= 2 && ZSIZE_SUBDOMAIN >= 2);
    
    if (rbcs)
    {
	CudaRBC::Extent host_extent;
	CudaRBC::setup(nvertices, host_extent);
    }
    
    MPI_CHECK(MPI_Comm_dup(_cartcomm, &cartcomm));
	    
    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
	    
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
	    
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

	//recvbufs[i].resize(nvertices * 10);
	//sendbufs[i].resize(nvertices * 10);
    }

    CUDA_CHECK(cudaEventCreate(&evextents, cudaEventDisableTiming));

    _post_recvcount();
}

void RedistributeRBCs::_compute_extents(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RDC/extent", NVTX_C7);

#if 1
    if (nrbcs)
	minmax(xyzuvw, nvertices, nrbcs, minextents.devptr, maxextents.devptr, stream);
#else
    for(int i = 0; i < nrbcs; ++i)
	CudaRBC::extent_nohost(stream, (float *)(xyzuvw + nvertices * i), extents.devptr + i);
#endif
}

namespace ReorderingRBC
{
    static const int cmaxnrbcs = 64 * 4;
    __constant__ float * csources[cmaxnrbcs], * cdestinations[cmaxnrbcs];

    template <bool from_cmem>
    __global__ void pack_all_kernel(const int nrbcs, const int nvertices, 
				    const float ** const dsources, float ** const ddestinations)
    {
	if (nrbcs == 0)
	    return;

	const int nfloats_per_rbc = 6 * nvertices;

	assert(nfloats_per_rbc * nrbcs <= blockDim.x * gridDim.x);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (gid >= nfloats_per_rbc * nrbcs) 
	    return;

	const int idrbc = gid / nfloats_per_rbc;
	assert(idrbc < nrbcs);

	const int offset = gid % nfloats_per_rbc;
	
	float val;
	if (from_cmem)
	    val = csources[idrbc][offset];
	else
	    val = dsources[idrbc][offset];
	
	if (from_cmem)
	    cdestinations[idrbc][offset] = val;
	else
	    ddestinations[idrbc][offset] = val;
    }

    SimpleDeviceBuffer<float *> *_ddestinations = new SimpleDeviceBuffer<float*>;
    SimpleDeviceBuffer<const float *> *_dsources = new SimpleDeviceBuffer<const float*>;

    void pack_all(cudaStream_t stream, const int nrbcs, const int nvertices, const float ** const sources, float ** const destinations)
    {
	if (nrbcs == 0)
	    return;

	const int nthreads = nrbcs * nvertices * 6;

	if (nrbcs < cmaxnrbcs)
	{
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(cdestinations, destinations, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
	    CUDA_CHECK(cudaMemcpyToSymbolAsync(csources, sources, sizeof(float *) * nrbcs, 0, cudaMemcpyHostToDevice, stream));
	    
	    pack_all_kernel<true><<<(nthreads + 127) / 128, 128, 0, stream>>>(nrbcs, nvertices, NULL, NULL);
	}
	else
	{
	    _ddestinations->resize(nrbcs);
	    _dsources->resize(nrbcs);

	    CUDA_CHECK(cudaMemcpyAsync(_ddestinations->data, destinations, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));
	    CUDA_CHECK(cudaMemcpyAsync(_dsources->data, sources, sizeof(float *) * nrbcs, cudaMemcpyHostToDevice, stream));

	    pack_all_kernel<false><<<(nthreads + 127) / 128, 128, 0, stream>>>(nrbcs, nvertices, _dsources->data, _ddestinations->data);
	}

	CUDA_CHECK(cudaPeekAtLastError());
    }
}

void RedistributeRBCs::extent(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RDC/extent", NVTX_C2);

    minextents.resize(nrbcs);
    maxextents.resize(nrbcs);

    CUDA_CHECK(cudaPeekAtLastError());

    _compute_extents(xyzuvw, nrbcs, stream);

    CUDA_CHECK(cudaPeekAtLastError());

    CUDA_CHECK(cudaEventRecord(evextents, stream));
}
    
void RedistributeRBCs::pack_sendcount(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RDC/pack-sendcount", NVTX_C3);

    CUDA_CHECK(cudaEventSynchronize(evextents));

    std::vector<int> reordering_indices[27];

    for(int i = 0; i < nrbcs; ++i)
    {
	const float3 minext = minextents.data[i];
	const float3 maxext = maxextents.data[i];

	float p[3] = {
	    0.5 * (minext.x + maxext.x),
	    0.5 * (minext.y + maxext.y),
	    0.5 * (minext.z + maxext.z)
	};
	
	const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

	int vcode[3];
	for(int c = 0; c < 3; ++c)
	    vcode[c] = (2 + (p[c] >= -L[c]/2) + (p[c] >= L[c]/2)) % 3;
	
	const int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);

	reordering_indices[code].push_back(i);
    }

    bulk.resize(reordering_indices[0].size() * nvertices);

    for(int i = 1; i < 27; ++i)
	halo_sendbufs[i].resize(reordering_indices[i].size() * nvertices);

#if 1
    {
	static std::vector<const float *> src;
	static std::vector<float *> dst;

	src.clear();
	dst.clear();

	for(int i = 0; i < 27; ++i)
	    for(int j = 0; j < reordering_indices[i].size(); ++j)
	    {
		src.push_back((float *)(xyzuvw + nvertices * reordering_indices[i][j]));
		
		if (i)
		    dst.push_back((float *)(halo_sendbufs[i].devptr + nvertices * j));
		else
		    dst.push_back((float *)(bulk.data + nvertices * j));
	    }
	
	ReorderingRBC::pack_all(stream, src.size(), nvertices, &src.front(), &dst.front());
	
	CUDA_CHECK(cudaPeekAtLastError());
    }
#else
    for(int j = 0; j < reordering_indices[0].size(); ++j)
	CUDA_CHECK(cudaMemcpyAsync(bulk.data + nvertices * j, xyzuvw + nvertices * reordering_indices[0][j],
				   sizeof(Particle) * nvertices, cudaMemcpyDeviceToDevice, stream));

    for(int i = 1; i < 27; ++i)
	for(int j = 0; j < reordering_indices[i].size(); ++j)
	    CUDA_CHECK(cudaMemcpyAsync(halo_sendbufs[i].devptr + nvertices * j, xyzuvw + nvertices * reordering_indices[i][j],
				       sizeof(Particle) * nvertices, cudaMemcpyDeviceToDevice, stream));
#endif

    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    for(int i = 1; i < 27; ++i)
	MPI_CHECK( MPI_Isend(&halo_sendbufs[i].size, 1, MPI_INTEGER, rankneighbors[i], i + 1024, cartcomm, &sendcountreq[i-1]) );
}

void RedistributeRBCs::_post_recvcount()
{
    recv_counts[0] = 0;

    for(int i = 1; i < 27; ++i)
    {
	MPI_Request req;

	MPI_CHECK( MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, anti_rankneighbors[i], i + 1024, cartcomm, &req) );
	
	recvcountreq.push_back(req);
    }
}

int RedistributeRBCs::post()
{
    NVTX_RANGE("RDC/post", NVTX_C3);

    {
	MPI_Status statuses[recvcountreq.size()];
	MPI_CHECK( MPI_Waitall(recvcountreq.size(), &recvcountreq.front(), statuses) );
	recvcountreq.clear();
    }

    arriving = 0;
    for(int i = 1; i < 27; ++i)
    {
	const int count = recv_counts[i];
	
	arriving += count;
	halo_recvbufs[i].resize(count);
    }
    
    arriving /= nvertices;
    notleaving = bulk.size / nvertices;
  
    MPI_Status statuses[26];	    
    MPI_CHECK( MPI_Waitall(26, sendcountreq, statuses) );

    for(int i = 1; i < 27; ++i)
	if (halo_recvbufs[i].size > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(halo_recvbufs[i].data, halo_recvbufs[i].size, Particle::datatype(),
				anti_rankneighbors[i], i + 1155, cartcomm, &request));

	    recvreq.push_back(request);
	}

    for(int i = 1; i < 27; ++i)
	if (halo_sendbufs[i].size > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Isend(halo_sendbufs[i].data, halo_sendbufs[i].size, Particle::datatype(),
				rankneighbors[i], i + 1155, cartcomm, &request));

	    sendreq.push_back(request);
	}

    return notleaving + arriving;
}

namespace ParticleReorderingRBC
{
    __global__ void shift(const Particle * const psrc, const int np, const int code, const int rank, 
			  const bool check, Particle * const pdst)
    {
	assert(blockDim.x * gridDim.x >= np);
	
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	
	int d[3] = { (code + 1) % 3 - 1, (code / 3 + 1) % 3 - 1, (code / 9 + 1) % 3 - 1 };
	
	if (pid >= np)
	    return;
	
#ifndef NDEBUG
	Particle old = psrc[pid];
#endif
	Particle pnew = psrc[pid];

	const int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};

	for(int c = 0; c < 3; ++c)
	    pnew.x[c] -= d[c] * L[c];

	pdst[pid] = pnew;

#ifndef NDEBUG
	if (check)
	{
	    int vcode[3];
	    for(int c = 0; c < 3; ++c)
		vcode[c] = (2 + (pnew.x[c] >= -L[c]/2) + (pnew.x[c] >= L[c]/2)) % 3;
		
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

void RedistributeRBCs::unpack(Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RDC/recv-unpack", NVTX_C7);

    assert(notleaving + arriving == nrbcs);

    MPI_Status statuses[26];
    MPI_CHECK(MPI_Waitall(recvreq.size(), &recvreq.front(), statuses) );
    MPI_CHECK(MPI_Waitall(sendreq.size(), &sendreq.front(), statuses) );
    
    recvreq.clear();
    sendreq.clear();
   
    CUDA_CHECK(cudaMemcpyAsync(xyzuvw, bulk.data, notleaving * nvertices * sizeof(Particle), 
			       cudaMemcpyDeviceToDevice, stream));
    
    for(int i = 1, s = notleaving * nvertices; i < 27; ++i)
    {
	const int count =  halo_recvbufs[i].size;

	if (count > 0)
	    ParticleReorderingRBC::shift<<< (count + 127) / 128, 128, 0, stream >>>
		(halo_recvbufs[i].devptr, count, i, myrank, false, xyzuvw + s);

	assert(s <= nrbcs * nvertices);

	s += halo_recvbufs[i].size;
    }

    CUDA_CHECK(cudaPeekAtLastError());

    _post_recvcount();
}

RedistributeRBCs::~RedistributeRBCs()
{    
    MPI_CHECK(MPI_Comm_free(&cartcomm));
}
