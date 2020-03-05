#pragma once

#include "utils/fragments_mapping.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>

#include <mpi.h>
#include <string>
#include <vector>

namespace mirheo
{


/** A device-compatible structure that holds buffer information for
    packing / unpacking data.
    In general, there is one buffer per source/destination rank.
    The current implementation uses a single array that contains all buffers in a contiguous way.
    The offsetsBytes values (one per buffer) state where each buffer start within the array.
 */
struct BufferOffsetsSizesWrap
{
    int nBuffers;  ///< number of buffers

    char *buffer;  ///< device data pointer to the array containing all buffers
    int *offsets;  ///< device array of size #nBuffers+1 with i-th buffer start index (in number of elements)
    int *sizes;    ///< device array of size #nBuffers with i-th buffer size (in number of elements)
    size_t *offsetsBytes; ///< device array of size #nBuffers+1 with i-th buffer start index (in number of bytes)

    /// \return buffer with id \p bufId
    __D__ inline char* getBuffer(int bufId)
    {
        return buffer + offsetsBytes[bufId];
    }
};

/// Structure held on the host only that contains pack/unpack buffers and their sizes/offsets (see BufferOffsetsSizesWrap).
struct BufferInfos
{
    PinnedBuffer<int> sizes;   ///< number of elements in each buffer
    PinnedBuffer<int> offsets; ///< prefix sum of the above
    PinnedBuffer<size_t> sizesBytes;   ///< number of bytes per buffer
    PinnedBuffer<size_t> offsetsBytes; ///< start of each buffer (in bytes) within the array
    PinnedBuffer<char> buffer; ///< all buffers in contiguous memory.
    std::vector<MPI_Request> requests; ///< send or recv requests associated to each buffer; only relevant for MPIExchangeEngine

    void clearAllSizes(cudaStream_t stream); ///< set sizes and sizesBytes to zero on host and device
    void resizeInfos(int nBuffers); ///< resize the size and offset buffers to support a given number of buffers
    void uploadInfosToDevice(cudaStream_t stream); ///< upload all size and offset information from host to device
    char* getBufferDevPtr(int bufId); ///< \return The device pointer to the buffer with the given index
};


class ParticlePacker;

/** \brief Manages communication data per ParticleVector.

    Each ExchangeEntity holds send and recv BufferInfos object for a given ParticleVector.
 */
class ExchangeEntity
{
public:
    /** \brief Construct an ExchangeEntity object
        \param name The name of the Corresponding ParticleVector
        \param uniqueId A positive integer. This must be unique when a collection of ExchangeEntity objects is registered in a single \c Exchanger.
        \param packer The class used to pack/unpack the data into buffers 
     */ 
    ExchangeEntity(std::string name, int uniqueId, ParticlePacker *packer);
    ~ExchangeEntity();

    /** \brief Compute the recv offsets and offsetsBytes on the host.
        \note the recv sizes must be available on the host.
     */
    void computeRecvOffsets();

    /** \brief Compute the send offsets and offsetsBytes on the host.
        \note the send sizes must be available on the host.
     */
    void computeSendOffsets();

    /** \brief Compute the send offsets and offsetBytes on the device and download all sizes and offsets on the host.
        \note The send sizes must be available on the device
     */
    void computeSendOffsets_Dev2Dev(cudaStream_t stream);

    void resizeSendBuf(); ///< resize the internal send buffers; requires send offsetsBytes to be available on the host
    void resizeRecvBuf(); ///< resize the internal recv buffers; requires recv offsetsBytes to be available on the host

    int getUniqueId() const; ///< \return the unique id
    
    BufferOffsetsSizesWrap wrapSendData(); ///< \return a BufferOffsetsSizesWrap from the send BufferInfos
    BufferOffsetsSizesWrap wrapRecvData(); ///< \return a BufferOffsetsSizesWrap from the recv BufferInfos

    const std::string& getName() const; ///< \return the name of the attached ParticleVector
    const char* getCName() const;       ///< \return the name of the attached ParticleVector in c-style string
    
public:
    const int nBuffers = fragment_mapping::numFragments; ///< equal to number of neighbours + 1 (for bulk)
    const int bulkId   = fragment_mapping::bulkId;       ///< The index of the bulk buffer
    
    BufferInfos send; ///< buffers for the send data
    BufferInfos recv; ///< buffers for the recv data
    std::vector<int> recvRequestIdxs; ///< only relevant for MPIExchangeEngine
    
private:
    std::string name_;       ///< corresponding ParticleVector name
    int uniqueId_;           ///< a unique exchange id: used for tags
    ParticlePacker *packer_; ///< Helper to pack the data into buffers
};

} // namespace mirheo
