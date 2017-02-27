#pragma once

#include "datatypes.h"
#include "containers.h"
#include "logger.h"
#include "celllist.h"

#include <vector>
//#include <thread>

struct HaloHelper
{
	PinnedBuffer<int> counts;
	PinnedBuffer<Particle> sendBufs[27];
	PinnedBuffer<float4*>  sendAddrs;
	HostBuffer<Particle> recvBufs[27];

	std::vector<MPI_Request> requests;

	cudaStream_t stream;
	//std::thread thread;
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

	cudaStream_t defStream;

	std::vector<std::pair<ParticleVector*, CellList*>> particlesAndCells;
	std::vector<HaloHelper> helpers;


	void send(int vid);
	void receive(int vid);

public:

	void _initialize(int vid);

	HaloExchanger(MPI_Comm& comm, cudaStream_t defStream);
	void attach(ParticleVector* pv, CellList* cl);
	void init();
	void finalize();
};
