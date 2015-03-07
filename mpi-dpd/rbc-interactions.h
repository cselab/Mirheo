#pragma once

#include <vector>
#include <rbc-cuda.h>

#include "common.h"

#include <../dpd-rng.h>

class ComputeInteractionsRBC
{
protected:

    MPI_Comm cartcomm;

    std::vector<MPI_Request> reqsendp, reqrecvp, reqsendacc, reqrecvacc;
    
    int nvertices, myrank, nranks, dims[3], periods[3], coords[3], dstranks[26], recv_tags[26], recv_counts[26], send_counts[26];

    std::vector< int > haloreplica[26];
    
    PinnedHostBuffer<CudaRBC::Extent> extents;

    struct
    {
	PinnedHostBuffer<Particle> state;
	PinnedHostBuffer<Acceleration> result;

	void setup(int n) { state.resize(n); result.resize(n); }
	
    } remote[26], local[26];

    void pack_and_post(const Particle * const rbcs, const int nrbcs);
    
    virtual void _compute_extents(const Particle * const xyzuvw, const int nrbcs);
    virtual void _internal_forces(const Particle * const xyzuvw, const int nrbcs, Acceleration * acc);

    void _wait(std::vector<MPI_Request>& v)
    {
	MPI_Status statuses[26];
	
	if (v.size())
	    MPI_CHECK(MPI_Waitall(v.size(), &v.front(), statuses));

	v.clear();
    }

    Logistic::KISS local_trunk;

public:

    cudaStream_t stream;

    ComputeInteractionsRBC(MPI_Comm cartcomm);
    
    void evaluate(const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
		  const int * const cellsstart_solvent, const int * const cellscount_solvent,
		  const Particle * const rbcs, const int nrbcs, Acceleration * accrbc);

    ~ComputeInteractionsRBC();
};
