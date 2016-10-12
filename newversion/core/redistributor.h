#pragma once

#include "datatypes.h"
#include "containers.h"
#include "logger.h"

#include <vector>
#include <thread>

// Note: we need to send / rcv accelarations as well here
// simply because integration is performed later on

struct PandA
{
	Particle p;
	Acceleration a;
};

struct RedistributorHelper
{
	PinnedBuffer<int> counts;
	PinnedBuffer<PandA> sendBufs[27];
	PinnedBuffer<float4*>  sendAddrs;
	HostBuffer<PandA> recvBuf;
	HostBuffer<Particle> recvPartBuf;
	HostBuffer<Acceleration> recvAccBuf;

	// Async should be beneficial for multiple redistributors
	cudaStream_t stream;
	std::thread thread;
};

class Redistributor
{
private:
	int dir2rank[27];
	int nActiveNeighbours;
	int myrank;
	MPI_Datatype mpiPandAType;
	MPI_Comm redComm;

	std::vector<ParticleVector*> particleVectors;
	std::vector<RedistributorHelper> helpers;


	void postReceive(int vid);
	void send(int vid);
	void receive(int vid);

public:

	// Has to be private, but cuda doesn't support lambdas in private functions
	void __identify(int vid, float dt);

	Redistributor(MPI_Comm& comm);
	void attach(ParticleVector* pv, int ndens);
	void redistribute(float dt);
};
