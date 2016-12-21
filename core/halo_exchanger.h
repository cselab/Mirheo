#pragma once

#include "datatypes.h"
#include "containers.h"
#include "logger.h"

#include <vector>
#include <thread>

struct HaloHelper
{
	PinnedBuffer<int> counts;
	PinnedBuffer<Particle> sendBufs[27];
	PinnedBuffer<float4*>  sendAddrs;
	HostBuffer<Particle> recvBufs[27];

	std::vector<MPI_Request> requests;

	cudaStream_t stream;
	std::thread thread;
};

class HaloExchanger
{
private:
	int dir2rank[27];
	int compactedDirs[26]; // nActiveNeighbours entries s.t. we need to send/recv to/from dir2rank[compactedDirs[i]], i=0..nActiveNeighbours

	int nActiveNeighbours;
	int myrank;
	MPI_Datatype mpiPartType;
	MPI_Comm haloComm;

	std::vector<ParticleVector*> particleVectors;
	std::vector<HaloHelper> helpers;


	void send(int vid);
	void receive(int vid);

public:

	void _initialize(int vid);

	HaloExchanger(MPI_Comm& comm);
	void attach(ParticleVector* pv, int ndens);
	void exchange();
};
