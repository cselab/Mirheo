#pragma once

#include <cassert>
#include <vector>
#include <mpi.h>

#include "common.h"

//see the vanilla version of this code for details about how this class operates
class ComputeInteractionsDPD
{   
    MPI_Comm cartcomm;
    MPI_Request sendreq[26], recvreq[26];

    //mpi send and recv informations
    int L, myrank, nranks, dims[3], periods[3], coords[3], dstranks[26], recv_tags[26];
    
    bool pending_send;

    //zero-copy allocation for acquiring the message offsets in the gpu send buffer
    int * sendpacks_start, * sendpacks_start_host, *send_bag_size_required, *send_bag_size_required_host;

    //plain copy of the offsets for the cpu (i speculate that reading multiple times the zero-copy entries is slower)
    int send_offsets[27], recv_offsets[27];

    //send and receive gpu buffer for receiving and sending halos
    Particle *send_bag, *recv_bag;
    int send_bag_size, recv_bag_size;

    //temporary buffer to compute accelerations in the halo
    Acceleration * acc_remote;

    //cuda-sync after to wait for packing of the halo
    void dpd_remote_interactions_stage1(const Particle * const p, const int n);

    //mpi-sync for the surrounding halos
    void dpd_remote_interactions_stage2(const Particle * const p, const int n, int saru_tag1, Acceleration * const a);

    cudaStream_t streams[7];
    int code2stream[26];
    
public:
    
    ComputeInteractionsDPD(MPI_Comm cartcomm, int L);
    
    ~ComputeInteractionsDPD();

    void evaluate(int& saru_tag, const Particle * const p, int n, Acceleration * const a,
		  const int * const cellsstart, const int * const cellscount);
};
