#pragma once

#include <cassert>
#include <vector>
#include <mpi.h>

#include "common.h"

class ComputeInteractionsDPD
{
    MPI_Comm cartcomm;
    MPI_Request sendreq[26], recvreq[26];

    int L, myrank, nranks, dims[3], periods[3], coords[3];
    bool send_pending;

    std::vector<Particle> mypacks[26], srcpacks[26];
    std::vector<int> myentries[26];
    
    void dpd_kernel(Particle * p, int n, int saru_tag,  Acceleration * a);
    
    void dpd_bipartite_kernel(Particle * pdst, int ndst, Particle * psrc, int nsrc,
			      int saru_tag1, int saru_tag2, int saru_mask, Acceleration * a);

    void dpd_remote_interactions_stage1(Particle * p, int n);
	    
    void dpd_remote_interactions_stage2(Particle * p, int n, int saru_tag1, Acceleration * a);
    
public:
    
ComputeInteractionsDPD(MPI_Comm cartcomm, int L):
    cartcomm(cartcomm), L(L), send_pending(false)
    {
	assert(L % 2 == 0);
	MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
		   
	MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
	MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));
    }
    
    void evaluate(int& saru_tag, Particle * p, int n, Acceleration * a)
    {
	dpd_remote_interactions_stage1(p, n);
	    
	dpd_kernel(p, n, saru_tag, a);
	saru_tag += nranks - myrank;
	 
	dpd_remote_interactions_stage2(p, n, saru_tag, a);
	saru_tag += 1 + myrank;  
    }
};
