#pragma once

#include <mpi.h>

#include "common.h"

class HaloExchanger
{
    MPI_Request sendreq[26], recvreq[26];

    //mpi send and recv informations
    int recv_tags[26], nlocal;
    
    bool pending_send;

protected:
    MPI_Comm cartcomm;
    int L, myrank, nranks, dims[3], periods[3], coords[3], dstranks[26];
    
    //zero-copy allocation for acquiring the message offsets in the gpu send buffer
    int * sendpacks_start, * sendpacks_start_host, *send_bag_size_required, *send_bag_size_required_host;

    //plain copy of the offsets for the cpu (i speculate that reading multiple times the zero-copy entries is slower)
    int send_offsets[27], recv_offsets[27];

    //send and receive gpu buffer for receiving and sending halos
    Particle *send_bag, *recv_bag;
    int send_bag_size, recv_bag_size, *scattered_entries;

    //cuda-sync after to wait for packing of the halo, mpi non-blocking
    void pack_and_post(const Particle * const p, const int n);

    //mpi-sync for the surrounding halos, shift particle coord to the sysref of this rank
    void wait_for_messages();
    
    int nof_sent_particles();
    
public:
    
    HaloExchanger(MPI_Comm cartcomm, int L);
    
    ~HaloExchanger();

    void exchange(const Particle * const plocal, int nlocal, const Particle *& premote, int& nremote);
};
