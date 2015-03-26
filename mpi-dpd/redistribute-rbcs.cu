/*
 *  redistribute-rbcs.cu
 *  Part of CTC/mpi-dpd/
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
#include "minmax-massimo.h"

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
}

void RedistributeRBCs::_compute_extents(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RDC/extent", NVTX_C7);

#if 1
    minmax_massimo(xyzuvw, nvertices, nrbcs, minextents.devptr, maxextents.devptr, stream);
#else
    for(int i = 0; i < nrbcs; ++i)
	CudaRBC::extent_nohost(stream, (float *)(xyzuvw + nvertices * i), extents.devptr + i);
#endif
}

int RedistributeRBCs::stage1(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RDC/stage1", NVTX_C3);

    //extents.resize(nrbcs);
    minextents.resize(nrbcs);
    maxextents.resize(nrbcs);

    _compute_extents(xyzuvw, nrbcs, stream);

    CUDA_CHECK(cudaEventRecord(evextents));
    CUDA_CHECK(cudaEventSynchronize(evextents));

    std::vector<int> reordering_indices[27];

    for(int i = 0; i < nrbcs; ++i)
    {
	//const CudaRBC::Extent ext = extents.data[i];
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

    for(int i = 0; i < 27; ++i)
	sendbufs[i].resize(reordering_indices[i].size() * nvertices);

    for(int i = 0; i < 27; ++i)
	for(int j = 0; j < reordering_indices[i].size(); ++j)
	    CUDA_CHECK(cudaMemcpyAsync(sendbufs[i].devptr + nvertices * j, xyzuvw + nvertices * reordering_indices[i][j],
				       sizeof(Particle) * nvertices, cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
//I need to post receive first
    MPI_Request sendcountreq[26];
    for(int i = 1; i < 27; ++i)
	MPI_CHECK( MPI_Isend(&sendbufs[i].size, 1, MPI_INTEGER, rankneighbors[i], i + 1024, cartcomm, &sendcountreq[i-1]) );

    arriving = 0;
    for(int i = 1; i < 27; ++i)
    {
	int count;
	
	MPI_Status status;
	MPI_CHECK( MPI_Recv(&count, 1, MPI_INTEGER, anti_rankneighbors[i], i + 1024, cartcomm, &status) );

	arriving += count;
	recvbufs[i].resize(count);
    }
    
    arriving /= nvertices;
    notleaving = sendbufs[0].size / nvertices;

    if (arriving)
	printf("YEE something is arriving to rank %d (arriving %d)\n", myrank, arriving);
  
    MPI_Status statuses[26];	    
    MPI_CHECK( MPI_Waitall(26, sendcountreq, statuses) );


    for(int i = 1; i < 27; ++i)
	if (recvbufs[i].size > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(recvbufs[i].data, recvbufs[i].size, Particle::datatype(),
				anti_rankneighbors[i], i + 1155, cartcomm, &request));

	    recvreq.push_back(request);
	}

    for(int i = 1; i < 27; ++i)
	if (sendbufs[i].size > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Isend(sendbufs[i].data, sendbufs[i].size, Particle::datatype(),
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

void RedistributeRBCs::stage2(Particle * const xyzuvw, const int nrbcs, cudaStream_t stream)
{
    NVTX_RANGE("RDC/stage2", NVTX_C7);

    assert(notleaving + arriving == nrbcs);

    MPI_Status statuses[26];
    MPI_CHECK(MPI_Waitall(recvreq.size(), &recvreq.front(), statuses) );
    MPI_CHECK(MPI_Waitall(sendreq.size(), &sendreq.front(), statuses) );
    
    recvreq.clear();
    sendreq.clear();
   
    CUDA_CHECK(cudaMemcpyAsync(xyzuvw, sendbufs[0].devptr, notleaving * nvertices * sizeof(Particle), 
			       cudaMemcpyDeviceToDevice, stream));
    
    for(int i = 1, s = notleaving * nvertices; i < 27; ++i)
    {
	const int count =  recvbufs[i].size;

	if (count > 0)
	    ParticleReorderingRBC::shift<<< (count + 127) / 128, 128, 0, stream >>>
		(recvbufs[i].devptr, count, i, myrank, false, xyzuvw + s);

	assert(s <= nrbcs * nvertices);

	s += recvbufs[i].size;
    }

    CUDA_CHECK(cudaPeekAtLastError());
}

RedistributeRBCs::~RedistributeRBCs()
{    
    MPI_CHECK(MPI_Comm_free(&cartcomm));
}