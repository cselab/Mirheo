#pragma once

#include <core/containers.h>

#include <mpi.h>

#include <vector>
#include <string>

struct ExchangeHelper
{
	int datumSize;
	const int nBuffers = 27;

	std::string name;

	std::vector<int> recvBufSizes, recvOffsets;
	std::vector<PinnedBuffer<char>> recvBufs;
	PinnedBuffer<char*> recvAddrs;

	PinnedBuffer<int> sendBufSizes;
	std::vector<PinnedBuffer<char>> sendBufs;
	PinnedBuffer<char*> sendAddrs;

	std::vector<MPI_Request> requests;

	ExchangeHelper(std::string name, const int datumSize = 0);

	void resizeSendBufs();
	void resizeRecvBufs();
	void setDatumSize(int size);
};

class ParticleExchanger
{
protected:
	int dir2rank[27], dir2sendTag[27], dir2recvTag[27];
	int nActiveNeighbours;

	int myrank;
	MPI_Comm haloComm;

	std::vector<ExchangeHelper*> helpers;

	int tagByName(std::string);

	void recv(ExchangeHelper* helper, cudaStream_t stream);
	void send(ExchangeHelper* helper, cudaStream_t stream);

	virtual void prepareData(int id, cudaStream_t stream) = 0;
	virtual void combineAndUploadData(int id, cudaStream_t stream) = 0;
	virtual bool needExchange(int id) = 0;

public:

	ParticleExchanger(MPI_Comm& comm);
	void init(cudaStream_t stream);
	void finalize(cudaStream_t stream);

	virtual ~ParticleExchanger() = default;
};
