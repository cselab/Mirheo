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

    int L, myrank, nranks, dims[3], periods[3], coords[3], dstranks[26], recv_tags[26];
    bool pending_send;

    int * sendpacks_start, * sendpacks_start_host, *send_bag_size_required, *send_bag_size_required_host;
    int send_bag_size, recv_bag_size, send_offsets[27], recv_offsets[27];

    Particle *send_bag, *recv_bag;
    Acceleration * acc_remote;
    
    void dpd_remote_interactions_stage1(Particle * p, int n);
	    
    void dpd_remote_interactions_stage2(Particle * p, int n, int saru_tag1, Acceleration * a);
    
public:
    
    ComputeInteractionsDPD(MPI_Comm cartcomm, int L);
    
    ~ComputeInteractionsDPD();
    
    //saru_tag is cumbersome, yet is at the core of the current dpd scheme.
    //see dpd-interactions.cpp for more details
    void evaluate(int& saru_tag, Particle * p, int n, Acceleration * a, int * cellsstart, int * cellscount);
};
