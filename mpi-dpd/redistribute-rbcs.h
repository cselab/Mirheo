#pragma once

#include <mpi.h>

#include <rbc-cuda.h>

#include "common.h"

class RedistributeRBCs
{
    MPI_Comm cartcomm;
    std::vector<MPI_Request> sendreq, recvreq;
    
    int L, myrank, dims[3], periods[3], coords[3], rankneighbors[27], anti_rankneighbors[27];

    PinnedHostBuffer /*SimpleDeviceBuffer*/<Particle> recvbufs[27], sendbufs[27];

    int nvertices, arriving, notleaving;

    PinnedHostBuffer<CudaRBC::Extent> extents;
    
public:

    cudaStream_t stream;
    
    RedistributeRBCs(MPI_Comm comm, const int L);
        
    int stage1(const Particle * const xyzuvw, const int nrbcs);
    
    void stage2(Particle * const xyzuvw, const int nrbcs);

    ~RedistributeRBCs();
};
