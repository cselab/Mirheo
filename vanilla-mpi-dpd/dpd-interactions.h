/*
 *  dpd-interactions.h
 *  Part of CTC/vanilla-mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-07.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <cassert>
#include <vector>
#include <mpi.h>

#include "common.h"

/* the acceleration of the particles is computed by considering two type of dpd interactions:
   - local interactions
   - interactions with remote particles, coming from the "halo" of the adjacent subdomains
   to achieve C/T overlap across the cluster i use the classical pattern:
   1. perform non-blocking send and receive request of the halo particles
   2. compute local interactions - hopefully this will overlap with step 1.
   3. compute remote interactions
*/
class ComputeInteractionsDPD
{
    MPI_Comm cartcomm;
    MPI_Request sendreq[26], recvreq[26];

    int L, myrank, nranks, dims[3], periods[3], coords[3];
    bool send_pending = false;

    std::vector<Particle> mypacks[26], srcpacks[26];
    std::vector<int> myentries[26];

    //compute the local interactions
    void dpd_kernel(Particle * p, int n, int saru_tag,  Acceleration * a);

    //compute the interactions between two distinct sets of particles
    //this is used to evaluate the remote interations
    void dpd_bipartite_kernel(Particle * pdst, int ndst, Particle * psrc, int nsrc,
			      int saru_tag1, int saru_tag2, int saru_mask, Acceleration * a);

    void dpd_remote_interactions_stage1(Particle * p, int n);
	    
    void dpd_remote_interactions_stage2(Particle * p, int n, int saru_tag1, Acceleration * a);
    
public:
    
ComputeInteractionsDPD(MPI_Comm cartcomm, int L):
    cartcomm(cartcomm), L(L)
    {
	assert(L % 2 == 0);
	MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
		   
	MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
	MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));
    }

    //saru_tag is cumbersome, yet is at the core of the current dpd scheme.
    //see dpd-interactions.cpp for more details
    void evaluate(int& saru_tag, Particle * p, int n, Acceleration * a)
    {
	dpd_remote_interactions_stage1(p, n);
	    
	dpd_kernel(p, n, saru_tag, a);
	saru_tag += nranks - myrank;
	 
	dpd_remote_interactions_stage2(p, n, saru_tag, a);
	saru_tag += 1 + myrank;  
    }
};
