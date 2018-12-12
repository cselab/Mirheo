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
    int datumSize;             ///< size in bytes on a single datum in a message, e.g. Particle size or size of packed object
    const int nBuffers = FragmentMapping::numFragments;   ///< equal to number of neighbours + 1, for now fixed
    const int bulkId   = FragmentMapping::bulkId;

    std::string name;  ///< corresponding ParticleVector name

    PinnedBuffer<int>  recvSizes;    ///< Number of received elements per each neighbour
    PinnedBuffer<int>  recvOffsets;  ///< Starting indices for i-th neighbour
    PinnedBuffer<char> recvBuf;      ///< Buffer keeping all the received data

    PinnedBuffer<int>  sendSizes;    ///< Number elements to send to each neighbour
    PinnedBuffer<int>  sendOffsets;  ///< Starting indices for i-th neighbour
    PinnedBuffer<char> sendBuf;      ///< Buffer keeping all the data needs to be sent

    std::vector<MPI_Request> requests;
    std::vector<int> reqIndex;

    ExchangeHelper(std::string name, const int datumSize = 0) :
        name(name), datumSize(datumSize)
    {
        recvSizes.  resize_anew(nBuffers);
        recvOffsets.resize_anew(nBuffers+1);

        sendSizes.  resize_anew(nBuffers);
        sendOffsets.resize_anew(nBuffers+1);
    }

    ~ExchangeHelper() = default;

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
    inline void computeRecvOffsets() { computeOffsets(recvSizes, recvOffsets); }

    /**
     * Compute #sendOffsets from #sendSizes by explicit scan.
     * Only use and update CPU part of the PinnedBuffer
     */
    inline void computeSendOffsets() { computeOffsets(sendSizes, sendOffsets); }

    /**
     * Compute #sendOffsets from #sendSizes by explicit scan.
     * Take GPU data from #sendSizes as input and download it
     * to the CPU, them update accordingly CPU and GPU data
     * of the #sendOffsets
     */
    inline void computeSendOffsets_Dev2Dev(cudaStream_t stream)
    {
        sendSizes.downloadFromDevice(stream);
        computeSendOffsets();
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
    void computeOffsets(const PinnedBuffer<int>& sz, PinnedBuffer<int>& of)
    {
        int n = sz.size();
        if (n == 0) return;

        of[0] = 0;
        for (int i=0; i < n; i++)
            of[i+1] = of[i] + sz[i];
    }
};
