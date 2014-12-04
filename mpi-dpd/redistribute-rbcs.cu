#include <vector>

#include "redistribute-particles.h"
#include "redistribute-rbcs.h"

RedistributeRBCs::RedistributeRBCs(MPI_Comm _cartcomm, const int L): L(L), nvertices(CudaRBC::get_nvertices()), stream(0)
{
    assert(L % 2 == 0);

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

	recvbufs[i].resize(nvertices * 10);
	sendbufs[i].resize(nvertices * 10);
    }
}

int RedistributeRBCs::stage1(const Particle * const xyzuvw, const int nrbcs)
{
    extents.resize(nrbcs);
 
    for(int i = 0; i < nrbcs; ++i)
	CudaRBC::extent_nohost(stream, (float *)(xyzuvw + nvertices * i), extents.devptr + i);

    CUDA_CHECK(cudaStreamSynchronize(stream));
   
    std::vector<int> reordering_indices[27];

    for(int i = 0; i < nrbcs; ++i)
    {
	const CudaRBC::Extent ext = extents.data[i];
	
	float p[3] = {
	    0.5 * (ext.xmin + ext.xmax),
	    0.5 * (ext.ymin + ext.ymax),
	    0.5 * (ext.zmin + ext.zmax)
	};
	
	int vcode[3];
	for(int c = 0; c < 3; ++c)
	    vcode[c] = (2 + (p[c] >= -L/2) + (p[c] >= L/2)) % 3;
	
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
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

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

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    for(int i = 1; i < 27; ++i)
	if (recvbufs[i].size > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(recvbufs[i].data, recvbufs[i].size, Particle::datatype(),
				anti_rankneighbors[i], i + 1155, cartcomm, &request));

	    recvreq.push_back(request);
	}

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError()); 

    for(int i = 1; i < 27; ++i)
	if (sendbufs[i].size > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Isend(sendbufs[i].data, sendbufs[i].size, Particle::datatype(),
				rankneighbors[i], i + 1155, cartcomm, &request));

	    sendreq.push_back(request);
	}
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

 
    //MPI_Status statuses[26];
    MPI_CHECK(MPI_Waitall(recvreq.size(), &recvreq.front(), statuses) );
    MPI_CHECK(MPI_Waitall(sendreq.size(), &sendreq.front(), statuses) );

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    return notleaving + arriving;
}

void RedistributeRBCs::stage2(Particle * const xyzuvw, const int nrbcs)
{
    assert(notleaving + arriving == nrbcs);
   

    CUDA_CHECK(cudaMemcpyAsync(xyzuvw, sendbufs[0].data, notleaving * nvertices * sizeof(Particle), cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    for(int i = 1, s = notleaving * nvertices; i < 27; ++i)
    {
	const int count =  recvbufs[i].size;

	if (count > 0)
	    ParticleReordering::shift<<< (count + 127) / 128, 128, 0, stream >>>(recvbufs[i].devptr, count, L, i, myrank, false, xyzuvw + s);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaPeekAtLastError());

	
	//printf("rank%d s: %d nrbcs: %d nvrtices: %d\n", myrank, s, nrbcs, nvertices);
	assert(s <= nrbcs * nvertices);

	s += recvbufs[i].size;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

}

RedistributeRBCs::~RedistributeRBCs()
{    
    MPI_CHECK(MPI_Comm_free(&cartcomm));
}