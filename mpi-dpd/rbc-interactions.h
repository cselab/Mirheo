/*
 *  rbc-interactions.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <vector>
#include <rbc-cuda.h>

#include "common.h"

#include <../dpd-rng.h>

class ComputeInteractionsRBC
{
protected:

    MPI_Comm cartcomm;
    MPI_Request reqsendcounts[26], reqrecvcounts[26];

    std::vector<MPI_Request> reqsendp, reqrecvp, reqsendacc, reqrecvacc;
    
    int nvertices, myrank, nranks, dims[3], periods[3], coords[3], dstranks[26], recv_tags[26], recv_counts[26], send_counts[26];

    std::vector< int > haloreplica[26];

    //PinnedHostBuffer<CudaRBC::Extent> extents;
    PinnedHostBuffer<float3> minextents, maxextents;

    struct
    {
	PinnedHostBuffer<Particle> state;
	PinnedHostBuffer<Acceleration> result;

	void setup(int n) { state.resize(n); result.resize(n); }
	
    } remote[26], local[26];

    //void pack_and_post(const Particle * const rbcs, const int nrbcs, cudaStream_t stream);
    
    virtual void _compute_extents(const Particle * const xyzuvw, const int nrbcs, cudaStream_t stream);
    

    void _wait(std::vector<MPI_Request>& v)
    {
	MPI_Status statuses[26];
	
	if (v.size())
	    MPI_CHECK(MPI_Waitall(v.size(), &v.front(), statuses));

	v.clear();
    }

    Logistic::KISS local_trunk;

    cudaEvent_t evextents, evfsi;

    CellLists dualcells;
    SimpleDeviceBuffer<Acceleration> lacc_solvent, lacc_solute;
    SimpleDeviceBuffer<Particle> reordered_solute;
    SimpleDeviceBuffer<int> reordering;
    HookedTexture texSolventStart, texSoluteStart, texSolvent, texSolute;

public:

    ComputeInteractionsRBC(MPI_Comm cartcomm);

    void extent(const Particle * const rbcs, const int nrbcs, cudaStream_t stream);
    void count(const int nrbcs);
    void exchange_count();
    void pack_p(const Particle * const rbcs, cudaStream_t stream);
    void post_p();

    virtual void internal_forces(const Particle * const xyzuvw, const int nrbcs, Acceleration * acc, cudaStream_t stream);

    void fsi_bulk(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
	     const int * const cellsstart_solvent, const int * const cellscount_solvent,
	     const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream);

    void fsi_halo(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
		  const int * const cellsstart_solvent, const int * const cellscount_solvent,
		  const Particle * const rbcs, const int nrbcs, Acceleration * accrbc, cudaStream_t stream);

    void post_a();

    void merge_a(Acceleration * accrbc, cudaStream_t stream);

    ~ComputeInteractionsRBC();
};
