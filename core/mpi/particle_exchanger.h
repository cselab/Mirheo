#pragma once

#include <core/containers.h>

#include <mpi.h>

#include <vector>
#include <string>


struct BufferOffsetsSizesWrap
{
	int nBuffers;

	char* buffer;
	int *offsets, *sizes;
};

class ExchangeHelper
{
public:
	int datumSize;
	const int nBuffers = 27;

	std::string name;

	PinnedBuffer<int>  recvSizes, recvOffsets;
	PinnedBuffer<char> recvBuf;

	PinnedBuffer<int>  sendSizes, sendOffsets;
	PinnedBuffer<char> sendBuf;

	std::vector<MPI_Request> requests;

	ExchangeHelper(std::string name, const int datumSize = 0);
	inline void setDatumSize(int size) { datumSize = size; }

	inline void makeRecvOffsets() { makeOffsets(recvSizes, recvOffsets); }
	inline void makeSendOffsets() { makeOffsets(sendSizes, sendOffsets); }
	inline void makeSendOffsets_Dev2Dev(cudaStream_t stream)
	{
		sendSizes.downloadFromDevice(stream);
		makeSendOffsets();
		sendOffsets.uploadToDevice(stream);
	}

	inline void resizeSendBuf() { sendBuf.resize_anew(sendOffsets[nBuffers] * datumSize); }
	inline void resizeRecvBuf() { recvBuf.resize_anew(recvOffsets[nBuffers] * datumSize); }

	inline BufferOffsetsSizesWrap wrapSendData()
	{
		return {nBuffers, sendBuf.devPtr(), sendOffsets.devPtr(), sendSizes.devPtr()};
	}

private:
	void makeOffsets(const PinnedBuffer<int>& sz, PinnedBuffer<int>& of);
};

class ParticleExchanger
{
public:
	ParticleExchanger(MPI_Comm& comm);
	void init(cudaStream_t stream);
	void finalize(cudaStream_t stream);

	virtual ~ParticleExchanger() = default;

protected:
	int dir2rank[27], dir2sendTag[27], dir2recvTag[27];
	int nActiveNeighbours;

	int myrank;
	MPI_Comm haloComm;

	std::vector<ExchangeHelper*> helpers;

	int tagByName(std::string);

	void postRecvSize(ExchangeHelper* helper);
	void recv(ExchangeHelper* helper, cudaStream_t stream);
	void send(ExchangeHelper* helper, cudaStream_t stream);

	virtual void prepareData(int id, cudaStream_t stream) = 0;
	virtual void combineAndUploadData(int id, cudaStream_t stream) = 0;
	virtual bool needExchange(int id) = 0;
};
