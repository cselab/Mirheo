/*
 *  redistribute-rbcs.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-01.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <mpi.h>

#include <rbc-cuda.h>

#include "common.h"

class RedistributeRBCs
{
protected:

    MPI_Comm cartcomm;
    MPI_Request sendcountreq[26];

    std::vector<MPI_Request> sendreq, recvreq, recvcountreq;
    
    int myrank, dims[3], periods[3], coords[3], rankneighbors[27], anti_rankneighbors[27];
    int recv_counts[27];

    SimpleDeviceBuffer<Particle> bulk;
    PinnedHostBuffer<Particle> halo_recvbufs[27], halo_sendbufs[27];

    int nvertices, arriving, notleaving;

    cudaEvent_t evextents;

    PinnedHostBuffer<float3> minextents, maxextents;

    virtual void _compute_extents(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream);

    void _post_recvcount();
    
public:
    
    RedistributeRBCs(MPI_Comm comm);
        
    void extent(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream);
    void pack_sendcount(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream);
    int post();
    void unpack(Particle * const xyzuvw, const int nrbcs, cudaStream_t stream);
    
    ~RedistributeRBCs();
};
