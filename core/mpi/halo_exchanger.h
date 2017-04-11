#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include <vector>
#include <string>

struct HaloHelper
{
	std::string name;

	PinnedBuffer<int>   counts;
	PinnedBuffer<char>  sendBufs[27];
	PinnedBuffer<char*> sendAddrs;
	PinnedBuffer<char>  recvBufs[27];
	PinnedBuffer<char*> recvAddrs;

	std::vector<int> recvOffsets;
	std::vector<MPI_Request> requests;

	PinnedBuffer<Particle>* halo;

	cudaStream_t stream;

	HaloHelper(std::string name, const int sizes[3], PinnedBuffer<Particle>* halo);
};

class HaloExchanger
{
protected:
	int dir2rank[27];
	int compactedDirs[27];
	int nActiveNeighbours;

	int myrank;
	MPI_Datatype mpiPartType, mpiForceType;
	MPI_Comm haloComm;

	cudaStream_t defStream;

	std::vector<HaloHelper*> helpers;

	void exchange(HaloHelper* helper, int typeSize);
	void uploadHalos(HaloHelper* helper);

public:

	virtual void _prepareHalos(int id) = 0;

	HaloExchanger(MPI_Comm& comm, cudaStream_t defStream);
	void init();
	void finalize();

	virtual ~HaloExchanger() = default;
};
