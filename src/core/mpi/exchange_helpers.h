#pragma once

#include "fragments_mapping.h"

#include <mpi.h>
#include <core/containers.h>
#include <string>
#include <vector>


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
    
    ExchangeHelper(std::string name, int uniqueId);

    ~ExchangeHelper();

    /**
     * Set the #datumSize. This is made as a separate function
     * because ParticleVectors may get additional data channels
     * dynamically during the simulation, therefore #datumSize
     * may need to change
     */
    void setDatumSize(int size);

    /**
     * Compute #recvOffsets from #recvSizes by explicit scan.
     * Only use and update CPU part of the PinnedBuffer
     */
    void computeRecvOffsets();

    /**
     * Compute #sendOffsets from #sendSizes by explicit scan.
     * Only use and update CPU part of the PinnedBuffer
     */
    void computeSendOffsets();

    /**
     * Compute #sendOffsets from #sendSizes by explicit scan.
     * Take GPU data from #sendSizes as input and download it
     * to the CPU, them update accordingly CPU and GPU data
     * of the #sendOffsets
     */
    void computeSendOffsets_Dev2Dev(cudaStream_t stream);

    void resizeSendBuf();
    void resizeRecvBuf();

    int getUniqueId() const;
    
    /**
     * Wrap GPU data from #sendBuf, #sendSizes and #sendOffsets
     * as well as nBuffers into a BufferOffsetsSizesWrap struct
     * that can be used in the device code.
     */
    BufferOffsetsSizesWrap wrapSendData();

public:
    const int nBuffers = FragmentMapping::numFragments;   ///< equal to number of neighbours + 1, for now fixed
    const int bulkId   = FragmentMapping::bulkId;

    int datumSize;                   ///< size in bytes on a single datum in a message, e.g. Particle size or size of packed object

    std::string name;                ///< corresponding ParticleVector name
    int uniqueId;                    ///< a unique exchange id: used for tags

    PinnedBuffer<int>  recvSizes;    ///< Number of received elements per each neighbour
    PinnedBuffer<int>  recvOffsets;  ///< Starting indices for i-th neighbour
    PinnedBuffer<char> recvBuf;      ///< Buffer keeping all the received data

    PinnedBuffer<int>  sendSizes;    ///< Number elements to send to each neighbour
    PinnedBuffer<int>  sendOffsets;  ///< Starting indices for i-th neighbour
    PinnedBuffer<char> sendBuf;      ///< Buffer keeping all the data needs to be sent

    std::vector<MPI_Request> recvRequests, sendRequests;
    std::vector<int> recvRequestIdxs;

private:
    void computeOffsets(const PinnedBuffer<int>& sz, PinnedBuffer<int>& of);
};
