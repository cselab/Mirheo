#pragma once

#include <core/containers.h>

#include <mpi.h>

#include <vector>
#include <string>
#include <map>


/// Structure with information about exchange buffers,
/// intended to be used on GPU only
struct BufferOffsetsSizesWrap
{
	int nBuffers;  ///< number of buffers

	char *buffer;  ///< device data pointer
	int *offsets;  ///< device array of size #nBuffers with i-th buffer start index
	int *sizes;    ///< device array of size #nBuffers with i-th buffer size
};


/**
 * Class that keeps communication data per ParticleVector.
 */
class ExchangeHelper
{
public:
	int datumSize;             ///< size in bytes on a single datum in a message, e.g. Particle size or size of packed object
	const int nBuffers = 27;   ///< equal to number of neighbours + 1, for now fixed

	std::string name;  ///< corresponding ParticleVector name

	PinnedBuffer<int>  recvSizes;    ///< Number of received elements per each neighbour
	PinnedBuffer<int>  recvOffsets;  ///< Starting indices for i-th neighbour
	PinnedBuffer<char> recvBuf;      ///< Buffer keeping all the received data

	PinnedBuffer<int>  sendSizes;    ///< Number elements to send to each neighbour
	PinnedBuffer<int>  sendOffsets;  ///< Starting indices for i-th neighbour
	PinnedBuffer<char> sendBuf;      ///< Buffer keeping all the data needs to be sent

	std::vector<MPI_Request> requests;
	std::vector<int> reqIndex;

	ExchangeHelper(std::string name, const int datumSize = 0);

	/**
	 * Set the #datumSize. This is made as a separate function
	 * because ParticleVectors may get additional data channels
	 * dynamically during the simulation, therefore #datumSize
	 * may need to change
	 */
	inline void setDatumSize(int size) { datumSize = size; }

	/**
	 * Compute #recvOffsets from #recvSizes by explicit scan.
	 * Only use and update CPU part of the PinnedBuffer
	 */
	inline void makeRecvOffsets() { makeOffsets(recvSizes, recvOffsets); }

	/**
	 * Compute #sendOffsets from #sendSizes by explicit scan.
	 * Only use and update CPU part of the PinnedBuffer
	 */
	inline void makeSendOffsets() { makeOffsets(sendSizes, sendOffsets); }

	/**
	 * Compute #sendOffsets from #sendSizes by explicit scan.
	 * Take GPU data from #sendSizes as input and download it
	 * to the CPU, them update accordingly CPU and GPU data
	 * of the #sendOffsets
	 */
	inline void makeSendOffsets_Dev2Dev(cudaStream_t stream)
	{
		sendSizes.downloadFromDevice(stream);
		makeSendOffsets();
		sendOffsets.uploadToDevice(stream);
	}

	inline void resizeSendBuf() { sendBuf.resize_anew(sendOffsets[nBuffers] * datumSize); }  ///< simple resize
	inline void resizeRecvBuf() { recvBuf.resize_anew(recvOffsets[nBuffers] * datumSize); }  ///< simple resize

	/**
	 * Wrap GPU data from #sendBuf, #sendSizes and #sendOffsets
	 * as well as nBuffers into a BufferOffsetsSizesWrap struct
	 * that can be used in the device code.
	 */
	inline BufferOffsetsSizesWrap wrapSendData()
	{
		return {nBuffers, sendBuf.devPtr(), sendOffsets.devPtr(), sendSizes.devPtr()};
	}

private:
	void makeOffsets(const PinnedBuffer<int>& sz, PinnedBuffer<int>& of);
};

/**
 * Base class implementing MPI exchange logic.
 *
 * The pipeline is as follows:
 * - base method init() sets up the communication:
 *   - calls base method postRecvSize() that issues MPI_Irecv() calls
 *     for sizes of data that has to be received
 *   - calls virtual method prepareData() that fills the corresponding
 *     ExchangeHelper buffers with the data to exchange
 * - base method finalize() runs the communication (it could
 *   be split into send/recv pair, maybe this will be done later):
 *   - calls base method send() that sends the data from ExchangeHelper
 *     buffers to the relevant MPI processes
 *   - calls base method recv() that blocks until the sizes of the
 *     data and data themselves are received and stored in the ExchangeHelper
 *   - calls virtual combineAndUploadData() that takes care
 *     of storing data from the ExchangeHelper to where is has to be
 */
class ParticleExchanger
{
public:
	ParticleExchanger(MPI_Comm& comm, bool gpuAwareMPI);
	void init(cudaStream_t stream);
	void finalize(cudaStream_t stream);

	virtual ~ParticleExchanger() = default;

protected:
	int dir2rank[27], dir2sendTag[27], dir2recvTag[27];
	int nActiveNeighbours;

	int myrank;
	MPI_Comm haloComm;
	bool gpuAwareMPI;
	int singleCopyThreshold = (1<<20);

	std::vector<ExchangeHelper*> helpers;

	int tagByName(std::string);

	void postRecvSize(ExchangeHelper* helper);
	void sendSizes(ExchangeHelper* helper);
	void postRecv(ExchangeHelper* helper);
	void wait(ExchangeHelper* helper, cudaStream_t stream);
	void send(ExchangeHelper* helper, cudaStream_t stream);

	/**
	 * This function has to provide data that has to be communicated in
	 * `helpers[id]`. It has to set `helpers[id].sendSizes` and
	 * `helpers[id].sendOffsets` on the CPU, but the bulk data of
	 * `helpers[id].sendBuf` must be only set on GPU. The reason is because
	 * in most cases offsets and sizes are anyways needed on CPU side
	 * to resize stuff, but bulk data are not; and it would be possible
	 * to change the MPI backend to CUDA-aware calls.
	 *
	 * @param id helper id that will be filled with data
	 */
	virtual void prepareSizes(int id, cudaStream_t stream) = 0;
	virtual void prepareData (int id, cudaStream_t stream) = 0;

	/**
	 * This function has to unpack the received data. Similarly to
	 * prepareData() function, when it is called `helpers[id].recvSizes`
	 * and `helpers[id].recvOffsets` are set according to the
	 * received data on the CPU only. However, `helpers[id].recvBuf`
	 * will contain already GPU data
	 *
	 * @param id helper id that is filled with the received data
	 */
	virtual void combineAndUploadData(int id, cudaStream_t stream) = 0;

	/**
	 * If the ParticleVector didn't change since the last similar MPI
	 * exchange, there is no need to run the exchange again. This function
	 * controls such behaviour
	 * @param id of the ParticleVector and associated ExchangeHelper
	 * @return true if exchange is required, false - if not
	 */
	virtual bool needExchange(int id) = 0;
};
