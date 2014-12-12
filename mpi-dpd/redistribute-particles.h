#pragma once

#include <mpi.h>

#include "common.h"

namespace ParticleReordering
{
    __global__ void shift(const Particle * const psrc, const int np, const int L, const int code,
			  const int rank, const bool check, Particle * const pdst);
}

//pls see the vanilla version of this code for the computational patterns of this class
class RedistributeParticles
{
    MPI_Comm cartcomm;

    bool pending_send;
    
    int L, myrank, dims[3], periods[3], coords[3], rankneighbors[27], anti_rankneighbors[27];
    int arriving_start[28], notleaving, arriving;

    //gpu buffer used as receive buffer of the 26 messages
    SimpleDeviceBuffer<Particle> reordered; 
    PinnedHostBuffer<Particle> sendbufs[27], recvbufs[27];
   
    //this is used for zero-copy retrieval of the mpi-send offsets of the 26 messages
    //within the gpu send buffer
    int * leaving_start,  *leaving_start_device;

    MPI_Request sendreq[27], recvreq[27];
    
    int recv_counts[27], send_counts[27];

public:

    cudaStream_t mystream;

    RedistributeParticles(MPI_Comm cartcomm, const int L);
    
    ~RedistributeParticles();
    
    //cuda-sync inside, before sending messages to other ranks
    int stage1(const Particle * const p, const int n);

    //mpi-sync inside to receive messages, no cuda-sync
    void stage2(Particle * const p, const int n);
};

