#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include <vector>
#include <string>

struct ExchangeHelper
{
	std::string name;

	PinnedBuffer<int>   counts;
	PinnedBuffer<char>  sendBufs[27];
	PinnedBuffer<char*> sendAddrs;
	PinnedBuffer<char>  recvBufs[27];
	PinnedBuffer<char*> recvAddrs;

	std::vector<int> recvOffsets;
	std::vector<MPI_Request> requests;

	char* target;

	cudaStream_t stream;

	ExchangeHelper(std::string name, const int sizes[3], PinnedBuffer<Particle>* halo);
};

class ParticleExchanger
{
protected:
	int dir2rank[27];
	int compactedDirs[27];
	int nActiveNeighbours;

	int myrank;
	MPI_Datatype mpiPartType, mpiForceType;
	MPI_Comm haloComm;

	cudaStream_t defStream;

	std::vector<ExchangeHelper*> helpers;

	void postRecv(ExchangeHelper* helper, int typeSize);
	void sendWait(ExchangeHelper* helper, int typeSize);

	void combineAndUpload(int id, int typeSize);

	virtual void prepareUploadTarget(int id) = 0;
	virtual void prepareData(int id) = 0;

public:

	ParticleExchanger(MPI_Comm& comm, cudaStream_t defStream);
	void init();
	void finalize();

	virtual ~ParticleExchanger() = default;
};
