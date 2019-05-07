#pragma once

#include "utils/fragments_mapping.h"

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
    int *offsets;  ///< device array of size #nBuffers+1 with i-th buffer start index in elements number
    int *sizes;    ///< device array of size #nBuffers with i-th buffer size in elements number
    size_t *offsetsBytes; ///< device array of size #nBuffers+1 with i-th buffer start index in bytes
};

struct BufferInfos
{
    PinnedBuffer<int> sizes, offsets;
    PinnedBuffer<size_t> sizesBytes, offsetsBytes;
    PinnedBuffer<char> buffer;
    std::vector<MPI_Request> requests;

    void clearAllSizes(cudaStream_t stream);
    void resizeInfos(int nBuffers);
};

class Packer;

/**
 * Class that keeps communication data per ParticleVector.
 */
class ExchangeHelper
{
public:
    
    ExchangeHelper(std::string name, int uniqueId, Packer *packer);

    ~ExchangeHelper();

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

    std::string name;                ///< corresponding ParticleVector name
    int uniqueId;                    ///< a unique exchange id: used for tags

    BufferInfos send, recv;
    std::vector<int> recvRequestIdxs;
    
private:
    Packer *packer;
};
