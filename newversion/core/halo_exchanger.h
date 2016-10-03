#pragma once

#include "datatypes.h"
#include "containers.h"
#include "logger.h"

#include <vector>
#include <thread>

struct HaloHelper
{
	PinnedBuffer<Particle> sides[6];
	PinnedBuffer<int> counts, limits;

	std::vector<Particle> sendBufs[27];

	cudaStream_t stream;
	std::thread thread;
};

class HaloExchanger
{
	int dir2rank[27];
	int nActiveNeighbours;
	int myrank;
	MPI_Datatype mpiParticleType;
	MPI_Comm haloComm;

	std::vector<ParticleVector*> particleVectors;
	std::vector<HaloHelper> helpers;


	void postReceive(int vid);
	void send(int vid);
	void receive(int vid);

public:

	HaloExchanger(MPI_Comm& comm);
	void attach(ParticleVector* pv);
	void exchangeInit();
	void exchangeFinalize();
};
