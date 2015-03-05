#pragma once

#include <mpi.h>

#include "common.h"
#include "scan.h"

class HaloExchanger
{
    MPI_Comm cartcomm;
    MPI_Request sendreq[26 * 2], recvreq[26], sendcellsreq[26], recvcellsreq[26], sendcountreq[26], recvcountreq[26];
    
    int recv_tags[26], recv_counts[26], nlocal, nactive;

    ScanEngine scan;

    bool firstpost;
    
protected:
    
    struct SendHalo
    {
	int expected;
	bool enabled;
	SimpleDeviceBuffer<int> scattered_entries, tmpstart, tmpcount, dcellstarts;
	SimpleDeviceBuffer<Particle> dbuf;
	PinnedHostBuffer<int> hcellstarts;
	PinnedHostBuffer<Particle> hbuf;

	void setup(const int estimate, const int nhalocells)
	    {
		adjust(estimate);
		scattered_entries.resize(estimate);
		dcellstarts.resize(nhalocells + 1);
		hcellstarts.resize(nhalocells + 1);
		tmpcount.resize(nhalocells + 1);
		tmpstart.resize(nhalocells + 1);
	    }

	void adjust(const int estimate)
	    {
		enabled = estimate > 0;
		expected = estimate;
		hbuf.resize(estimate);
		dbuf.resize(estimate);
	    }

    } sendhalos[26];

    struct RecvHalo
    {
	int expected;
	bool enabled;
	PinnedHostBuffer<int> hcellstarts;
	PinnedHostBuffer<Particle> hbuf;
	SimpleDeviceBuffer<Particle> dbuf;
	SimpleDeviceBuffer<int> dcellstarts;

	void setup(const int estimate, const int nhalocells)
	    {
		adjust(estimate);
		dcellstarts.resize(nhalocells + 1);
		hcellstarts.resize(nhalocells + 1);
	    }

	void adjust(const int estimate)
	    {
		enabled = estimate > 0;
		expected = estimate;
		hbuf.resize(estimate);
		dbuf.resize(estimate);
	    }

    } recvhalos[26];
      
    int myrank, nranks, dims[3], periods[3], coords[3], dstranks[26];
    
    //zero-copy allocation for acquiring the message offsets in the gpu send buffer
    int * required_send_bag_size, * required_send_bag_size_host;
        
    //plain copy of the offsets for the cpu (i speculate that reading multiple times the zero-copy entries is slower)
    int nsendreq;

    int3 halosize[26];
    float safety_factor;

    void post_expected_recv();
    
    //cuda-sync after to wait for packing of the halo, mpi non-blocking
    void pack_and_post(const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount);

    //mpi-sync for the surrounding halos, shift particle coord to the sysref of this rank
    void wait_for_messages();

    virtual void spawn_local_work() { }
    
    int nof_sent_particles();

    cudaStream_t streams[7];
    int code2stream[26];
     
    const int basetag;

    void _cancel_recv();

public:
    
    HaloExchanger(MPI_Comm cartcomm, const int basetag);

    void adjust_message_sizes(ExpectedMessageSizes sizes);
    
    virtual ~HaloExchanger();
};
