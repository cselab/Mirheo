#pragma once

#include "datatypes.h"
#include "containers.h"
#include "logger.h"
#include "iniparser.h"

#include <vector>
#include <thread>

// Note: we need to send / rcv accelarations as well here
// simply because integration is performed later on


struct RedistributorHelper
{
	PinnedBuffer<int> counts;
	PinnedBuffer<Particle> sendBufs[27];
	PinnedBuffer<float4*>  sendAddrs;
	HostBuffer<Particle> recvBufs[27];

	std::vector<MPI_Request> requests;

	// Async should be beneficial for multiple redistributors
	cudaStream_t stream;
	std::thread thread;
};

class Redistributor
{
private:
	int dir2rank[27];
	int compactedDirs[26]; // nActiveNeighbours entries s.t. we need to send/recv to/from dir2rank[compactedDirs[i]], i=0..nActiveNeighbours

	int nActiveNeighbours;
	int myrank;
	MPI_Datatype mpiPartType;
	MPI_Comm redComm;

	std::vector<ParticleVector*> particleVectors;
	std::vector<RedistributorHelper> helpers;


	void postReceive(int vid);
	void send(int vid);
	void receive(int vid);

	IniParser& config;

public:

	// Has to be private, but cuda doesn't support lambdas in private functions
	void _initialize(int vid, float dt);

	Redistributor(MPI_Comm& comm, IniParser& config);
	void attach(ParticleVector* pv, int ndens);
	void redistribute(float dt);
};
