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
    std::vector<MPI_Request> sendreq, recvreq;
    
    int myrank, dims[3], periods[3], coords[3], rankneighbors[27], anti_rankneighbors[27];

    SimpleDeviceBuffer<Particle> bulk;
    PinnedHostBuffer<Particle> halo_recvbufs[27], halo_sendbufs[27];

    int nvertices, arriving, notleaving;

    cudaEvent_t evextents;

    PinnedHostBuffer<float3> minextents, maxextents;

    virtual void _compute_extents(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream);
    
public:
    
    RedistributeRBCs(MPI_Comm comm);
        
    int stage1(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream);
    
    void stage2(Particle * const xyzuvw, const int nrbcs, cudaStream_t stream);

    ~RedistributeRBCs();
};
