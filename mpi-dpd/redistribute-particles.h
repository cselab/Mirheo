#pragma once

#include <mpi.h>

#include "common.h"

class RedistributeParticles
{
    static const int basetag = 950;
    
public:

    struct UnpackBuffer
    {
	float * buffer;
	int capacity;
    };
   
    struct PackBuffer : UnpackBuffer
    {
	int * scattered_indices;
    };

    int stage1(const Particle * const p, const int n, cudaStream_t stream);

    void stage2(Particle * const particles, const int nparticles, cudaStream_t);
   
    RedistributeParticles(MPI_Comm cartcomm);

    void adjust_message_sizes(ExpectedMessageSizes sizes);

    ~RedistributeParticles();
   
    int pack_size(const int code) { return send_sizes[code]; }
   
    float pinned_data(const int code, const int entry) { return pinnedhost_sendbufs[code][entry]; }
   
private:

    MPI_Comm cartcomm;

    bool firstcall;
    
    int dims[3], periods[3], coords[3], neighbor_ranks[27], recv_tags[27],
	default_message_sizes[27], send_sizes[27], recv_sizes[27],
	nsendmsgreq, nexpected, nbulk, nhalo;

    float safety_factor;

    int nactiveneighbors;

    MPI_Request sendcountreq[27], recvcountreq[27], sendmsgreq[27 * 2], recvmsgreq[27 * 2];

    cudaEvent_t evpacking, evsizes; //, evcompaction;

    void _waitall(MPI_Request * reqs, const int n)
    {
	MPI_Status statuses[n];
	MPI_CHECK( MPI_Waitall(n, reqs, statuses) );    
    }
   
    void _post_recv();
    void _cancel_recv();

    void _adjust_send_buffers(const int capacities[27]);
    void _adjust_recv_buffers(const int capacities[27]);
   
    PinnedHostBuffer<bool> failure; 
    PinnedHostBuffer<int> packsizes;
   
    float * pinnedhost_sendbufs[27], * pinnedhost_recvbufs[27];
   
    PackBuffer packbuffers[27];
    UnpackBuffer unpackbuffers[27];
};

