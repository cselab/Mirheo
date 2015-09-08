/*
 *  halo-exchanger.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-18.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <mpi.h>

#include "common.h"

class SolventExchange
{
    MPI_Comm cartcomm;
    MPI_Request sendreq[26 * 2], recvreq[26], sendcellsreq[26], recvcellsreq[26], sendcountreq[26], recvcountreq[26];
    
    int recv_tags[26], recv_counts[26], nlocal, nactive;

    bool firstpost;
    
protected:
    
    struct SendHalo
    {
	int expected;

	SimpleDeviceBuffer<int> scattered_entries, tmpstart, tmpcount, dcellstarts;
	SimpleDeviceBuffer<Particle> dbuf;
	PinnedHostBuffer<int> hcellstarts;
	PinnedHostBuffer<Particle> hbuf;

	void setup(const int estimate, const int nhalocells)
	    {
		adjust(estimate);
		dcellstarts.resize(nhalocells + 1);
		hcellstarts.resize(nhalocells + 1);
		tmpcount.resize(nhalocells + 1);
		tmpstart.resize(nhalocells + 1);
	    }

	void adjust(const int estimate)
	    {
		expected = estimate;
		hbuf.resize(estimate);
		dbuf.resize(estimate);
		scattered_entries.resize(estimate);
	    }

    } sendhalos[26];

    struct RecvHalo
    {
	int expected;

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

    void _pack_all(const Particle * const p, const int n, const bool update_baginfos, cudaStream_t stream);
    
    int nof_sent_particles();

    cudaEvent_t evfillall, evuploaded, evdownloaded;
     
    const int basetag;

    void _cancel_recv();

public:
    
    SolventExchange(MPI_Comm cartcomm, const int basetag);

    void pack(const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount, cudaStream_t stream);

    void post(const Particle * const p, const int n, cudaStream_t stream, cudaStream_t downloadstream);

    void recv(cudaStream_t stream, cudaStream_t uploadstream);

    void adjust_message_sizes(ExpectedMessageSizes sizes);

    virtual ~SolventExchange();
};
