/*
 *  rbc-interactions.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <vector>
#include <rbc-cuda.h>

#include "common.h"

#include <../dpd-rng.h>

class ComputeFSI
{
    enum { TAGBASE_C = 113, TAGBASE_P = 365, TAGBASE_A = 668, TAGBASE_P2 = 1055, TAGBASE_A2 = 1501 };

protected:

    MPI_Comm cartcomm;

    int iterationcount;

    int nranks, dstranks[26],
	dims[3], periods[3], coords[3], myrank,
	recv_tags[26], recv_counts[26], send_counts[26];

    cudaEvent_t evPpacked, evAcomputed;

    SimpleDeviceBuffer<Particle> packbuf;
    PinnedHostBuffer<Particle> host_packbuf;
    PinnedHostBuffer<int> requiredpacksizes, packstarts_padded;

    std::vector<MPI_Request> reqsendC, reqrecvC, reqsendP, reqrecvP, reqsendA, reqrecvA;

    Logistic::KISS local_trunk;

    class TimeSeriesWindow
    {
	static const int N = 100;

	int count, start, data[N];

    public:

    TimeSeriesWindow(): count(0), start(0) { }

	void update(int val)
	    {
		data[start] = ::max(0, val);
		start = (start + 1) % N;

		if (count < N)
		    ++count;
	    }

	int max() const
	    {
		int retval = 0;

		for(int i = 0; i < count; ++i)
		    retval = ::max(data[i], retval);

		return retval;
	    }

    };

    class RemoteHalo
    {
	TimeSeriesWindow history;

    public:

	SimpleDeviceBuffer<Particle> dstate;
	PinnedHostBuffer<Particle> hstate;
	PinnedHostBuffer<Acceleration> result;

	void preserve_resize(int n)
	    {
		dstate.resize(n);
		hstate.preserve_resize(n);
		result.resize(n);
		history.update(n);
	    }

	int expected() const { return history.max(); }

	int capacity() const { assert(hstate.capacity == dstate.capacity); return dstate.capacity; }

    } remote[26];

    class LocalHalo
    {
	TimeSeriesWindow history;

    public:

	SimpleDeviceBuffer<int> scattered_indices;
	PinnedHostBuffer<Acceleration> result;

	void resize(int n)
	{
	    scattered_indices.resize(n);
	    result.resize(n);
	}

	void update() { history.update(result.size); }

	int expected() const { return history.max(); }

	int capacity() const { return scattered_indices.capacity; }

    } local[26];

    void _adjust_packbuffers()
    {
	int s = 0;

	for(int i = 0; i < 26; ++i)
	    s += 32 * ((local[i].capacity() + 31) / 32);

	packbuf.resize(s);
	host_packbuf.resize(s);
    }

    void _wait(std::vector<MPI_Request>& v)
    {
	MPI_Status statuses[v.size()];

	if (v.size())
	    MPI_CHECK(MPI_Waitall(v.size(), &v.front(), statuses));

	v.clear();
    }

    void _postrecvC()
    {
	for(int i = 0; i < 26; ++i)
	{
	    MPI_Request reqC;

	    MPI_CHECK( MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i],
				 TAGBASE_C + recv_tags[i], cartcomm,  &reqC) );

	    reqrecvC.push_back(reqC);
	}
    }

    void _postrecvP()
    {
	for(int i = 0; i < 26; ++i)
	{
	    MPI_Request reqP;

	    MPI_CHECK( MPI_Irecv(remote[i].hstate.data, remote[i].expected() * 6, MPI_FLOAT, dstranks[i],
				 TAGBASE_P + recv_tags[i], cartcomm, &reqP) );

	    reqrecvP.push_back(reqP);
#if 0
	    if (iterationcount % 200 == 1 && myrank == 0)
	    {
		printf("exchange rate: S: %d R: %d\n", local[i].expected(), remote[i].expected());

		if (i == 25)
		    printf("========================================================\n");
	    }
#endif
	}
    }

    void _postrecvA()
    {
	for(int i = 0; i < 26; ++i)
	{
	    MPI_Request reqA;

	    MPI_CHECK( MPI_Irecv(local[i].result.data, local[i].result.size * 3, MPI_FLOAT, dstranks[i],
				 TAGBASE_A + recv_tags[i], cartcomm, &reqA) );

	    reqrecvA.push_back(reqA);
	}
    }

public:

    ComputeFSI(MPI_Comm cartcomm);

    void pack_p(const Particle * const solute, const int nsolute, cudaStream_t stream);

    void post_p(const Particle * const solute, const int nsolute, cudaStream_t stream, cudaStream_t downloadstream);

    void bulk(const Particle * const solvent, const int nsolvent, Acceleration * accsolvent,
	      const int * const cellsstart_solvent, const int * const cellscount_solvent,
	      const Particle * const solute, const int nrbcs, Acceleration * accsolute, cudaStream_t stream);

    void halo(const Particle * const solvent, const int nsolvent, Acceleration * accsolvent,
	      const int * const cellsstart_solvent, const int * const cellscount_solvent,
	      cudaStream_t stream, cudaStream_t uploadstream);

    void post_a();

    void merge_a(Acceleration * accsolute, const int nsolute, cudaStream_t stream);

    ~ComputeFSI();
};
