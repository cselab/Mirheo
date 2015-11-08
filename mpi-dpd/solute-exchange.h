/*
 *  solute-exchange.h
 *  Part of uDeviceX/mpi-dpd/
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

#include "common.h"

class SoluteExchange
{
    enum { TAGBASE_C = 113, TAGBASE_P = 365, TAGBASE_A = 668, TAGBASE_P2 = 1055, TAGBASE_A2 = 1501 };

public:

    struct Visitor { virtual void halo(ParticlesWrap allhalos, cudaStream_t stream) = 0; };

protected:

    MPI_Comm cartcomm;

    int iterationcount;

    int nranks, dstranks[26],
	dims[3], periods[3], coords[3], myrank,
	recv_tags[26], recv_counts[26], send_counts[26];

    cudaEvent_t evPpacked, evAcomputed, evAdownloaded;

    SimpleDeviceBuffer<int> packscount, packsstart, packsoffset, packstotalstart;
    PinnedHostBuffer<int> host_packstotalstart, host_packstotalcount;

    SimpleDeviceBuffer<Particle> packbuf;
    PinnedHostBuffer<Particle> host_packbuf;

    std::vector<ParticlesWrap> wsolutes;

    std::vector<MPI_Request> reqsendC, reqrecvC, reqsendP, reqrecvP, reqsendA, reqrecvA;

    std::vector<Visitor *> visitors;

    class TimeSeriesWindow
    {
	static const int N = 200;

	int count, data[N];

    public:

    TimeSeriesWindow(): count(0) { }

	void update(int val) { data[count++ % N] = ::max(0, val); }

	int max() const
	    {
		int retval = 0;

		for(int i = 0; i < min(N, count); ++i)
		    retval = ::max(data[i], retval);

		return retval;
	    }
    };

    class RemoteHalo
    {
	TimeSeriesWindow history;

    public:

	PinnedHostBuffer<Particle> hstate;
	PinnedHostBuffer<Acceleration> result;
	std::vector<Particle> pmessage;

	void preserve_resize(int n)
	    {
		hstate.preserve_resize(n);
		result.resize(n);
		history.update(n);
	    }

	int expected() const { return (int)ceil(history.max() * 1.1); }

	int capacity() const { return hstate.capacity; }

    } remote[26];

    SimpleDeviceBuffer<Particle> allremotehalos;
    SimpleDeviceBuffer<Acceleration> allremotehalosacc;

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

	int expected() const { return (int)ceil(history.max() * 1.1); }

	int capacity() const { assert(result.capacity == scattered_indices.capacity); return scattered_indices.capacity; }

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
#ifndef NDEBUG
	memset(recv_counts, 0x8f, sizeof(int) * 26);
#endif
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

#ifndef NDEBUG
	    memset(remote[i].hstate.data, 0xff, remote[i].hstate.capacity * sizeof(Particle));
#endif

	    remote[i].pmessage.resize(remote[i].expected());

	    MPI_CHECK( MPI_Irecv(&remote[i].pmessage.front(), remote[i].expected() * 6, MPI_FLOAT, dstranks[i],
				 TAGBASE_P + recv_tags[i], cartcomm, &reqP) );

	    reqrecvP.push_back(reqP);
	}
    }

    void _postrecvA()
    {
	for(int i = 0; i < 26; ++i)
	{
	    MPI_Request reqA;

#ifndef NDEBUG
	    memset(local[i].result.data, 0xff, local[i].result.capacity * sizeof(Acceleration));
#endif
	    MPI_CHECK( MPI_Irecv(local[i].result.data, local[i].result.size * 3, MPI_FLOAT, dstranks[i],
				 TAGBASE_A + recv_tags[i], cartcomm, &reqA) );

	    reqrecvA.push_back(reqA);
	}
    }

    void _not_nan(const float * const x, const int n) const
    {
#ifndef NDEBUG
	for(int i = 0; i < n; ++i)
	    assert(!isnan(x[i]));
#endif
    }

    void _pack_attempt(cudaStream_t stream);

public:

    SoluteExchange(MPI_Comm cartcomm);

    void bind_solutes(std::vector<ParticlesWrap> wsolutes) { this->wsolutes = wsolutes; }

    void attach_halocomputation(Visitor& visitor) { visitors.push_back(&visitor); }

    void pack_p(cudaStream_t stream);

    void post_p(cudaStream_t stream, cudaStream_t downloadstream);

    void recv_p(cudaStream_t uploadstream, cudaStream_t computestream);

    void halo(cudaStream_t uploadstream, cudaStream_t computestream, cudaStream_t downloadstream);

    void post_a();

    void recv_a(cudaStream_t stream);

    ~SoluteExchange();
};
