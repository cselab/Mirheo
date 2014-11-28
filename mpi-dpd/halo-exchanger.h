#pragma once

#include <mpi.h>

#include "common.h"
#include "scan.h"

class HaloExchanger
{
    MPI_Comm cartcomm;
    MPI_Request sendreq[26 * 2], recvreq[26], sendcellsreq[26], recvcellsreq[26], sendcountreq[26], recvcountreq[26];
    
    int recv_tags[26], recv_counts[26], nlocal;

    ScanEngine scan;
    
protected:
    
    struct SendHalo
    {
	int expected;
	SimpleDeviceBuffer<int> scattered_entries, cellstarts, tmpstart, tmpcount;
	SimpleDeviceBuffer<Particle> buf;
    } sendhalos[26];

    struct RecvHalo
    {
	int expected;
	SimpleDeviceBuffer<int> cellstarts;
	SimpleDeviceBuffer<Particle> buf;
    } recvhalos[26];
    
   
    int L, myrank, nranks, dims[3], periods[3], coords[3], dstranks[26];
    
    //zero-copy allocation for acquiring the message offsets in the gpu send buffer
    int * required_send_bag_size, * required_send_bag_size_host;
        
    //plain copy of the offsets for the cpu (i speculate that reading multiple times the zero-copy entries is slower)
    int nsendreq;

    int3 halosize[26];
    
    //cuda-sync after to wait for packing of the halo, mpi non-blocking
    void pack_and_post(const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount);

    //mpi-sync for the surrounding halos, shift particle coord to the sysref of this rank
    void wait_for_messages();
    
    int nof_sent_particles();

    cudaStream_t streams[7];
    int code2stream[26];
     
public:
    
    HaloExchanger(MPI_Comm cartcomm, int L);
    
    ~HaloExchanger();

    SimpleDeviceBuffer<Particle> exchange(const Particle * const plocal, int nlocal, const int * const cellsstart, const int * const cellscount);
};
