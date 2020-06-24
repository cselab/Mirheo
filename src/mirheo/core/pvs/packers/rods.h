// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "objects.h"

namespace mirheo
{

class LocalRodVector;

/** \brief A packer specific to rods.
    Will store particle, object and bisegment data into a single buffer.
 */
struct RodPackerHandler : public ObjectPackerHandler
{
    int nBisegments; ///< number of bisegment per rod
    GenericPackerHandler bisegments; ///< packer responsible for the bisegment data

    /// Get the reuired size (in bytes) of the buffer to hold the packed data
    __D__ size_t getSizeBytes(int numElements) const
    {
        return ObjectPackerHandler::getSizeBytes(numElements) +
            bisegments.getSizeBytes(numElements * nBisegments);
    }

#ifdef __CUDACC__
    /** \brief Fetch a full rod from the registered channels and pack it into the buffer.
        \param [in] numElements Number of rods that will be packed in the buffer.
        \param [out] buffer Destination buffer that will hold the packed rod
        \param [in] srcObjId The index of the rod to fetch from registered channels
        \param [in] dstObjId The index of the rod to store into the buffer
        \return The size (in bytes) taken by the packed data (numElements rods). Only relevant for thread with Id 0.

        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockPack(int numElements, char *buffer,
                                int srcObjId, int dstObjId) const
    {
        return _blockApply<PackOp>({}, numElements, buffer, srcObjId, dstObjId);
    }

    /** \brief Fetch a full rod from the registered channels, shift it and pack it into the buffer.
        \param [in] numElements Number of rods that will be packed in the buffer.
        \param [out] buffer Destination buffer that will hold the packed rod
        \param [in] srcObjId The index of the rod to fetch from registered channels
        \param [in] dstObjId The index of the rod to store into the buffer
        \param [in] shift The coordnate shift
        \return The size (in bytes) taken by the packed data (numElements rods). Only relevant for thread with Id 0.

        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockPackShift(int numElements, char *buffer,
                                     int srcObjId, int dstObjId, real3 shift) const
    {
        return _blockApply<PackShiftOp>({shift}, numElements, buffer, srcObjId, dstObjId);
    }

    /** \brief Unpack a full rod from the buffer and store it into the registered channels.
        \param [in] numElements Number of rods that will be packed in the buffer.
        \param [out] buffer Buffer that holds the packed rod
        \param [in] srcObjId The index of the rod to fetch from the buffer
        \param [in] dstObjId The index of the rod to store into the registered channels
        \return The size (in bytes) taken by the packed data (numElements objects). Only relevant for thread with Id 0.

        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockUnpack(int numElements, const char *buffer,
                                  int srcObjId, int dstObjId) const
    {
        return _blockApply<UnpackOp>({}, numElements, buffer, srcObjId, dstObjId);
    }

    /** \brief Unpack a full rod from the buffer and add it into the registered channels.
        \param [in] numElements Number of rods that will be packed in the buffer.
        \param [out] buffer Buffer that holds the packed rod
        \param [in] srcObjId The index of the rod to fetch from the buffer
        \param [in] dstObjId The index of the rod to store into the registered channels
        \param [in] eps Threshold under which the data will not be added
        \return The size (in bytes) taken by the packed data (numElements objects). Only relevant for thread with Id 0.

        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockUnpackAddNonZero(int numElements, const char *buffer,
                                            int srcObjId, int dstObjId, real eps) const
    {
         return _blockApply<UnpackAddOp>({eps}, numElements, buffer, srcObjId, dstObjId);
    }

protected:
    /** \brief Apply an operation on all data of a single rod.
        \tparam Operation The operation to apply to the datum
        \tparam BuffType Must be const char* or char* depending on the constness required by the operation
        \param [in] op The operation functor
        \param [in] numElements Number of objects to pack
        \param buffer Source or destination buffer, depending on op
        \param [in] srcObjId The index of the rod to fetch
        \param [in] dstObjId The index of the rod to store
        \return The size (in bytes) taken by the packed data (numElements rod). Only relevant for thread with Id 0.

        This method expects to be executed by a full CUDA block per rod.
     */
    template <class Operation, typename BuffType>
    __device__ size_t _blockApply(Operation op, int numElements, BuffType buffer,
                                  int srcObjId, int dstObjId) const
    {
        const int tid = threadIdx.x;
        size_t offsetBytes = ObjectPackerHandler::_blockApply(op, numElements, buffer,
                                                              srcObjId, dstObjId);

        buffer += offsetBytes;
        size_t ob = 0;

        for (int bid = tid; bid < nBisegments; bid += blockDim.x)
        {
            const int srcBid = srcObjId * nBisegments + bid;
            const int dstBid = dstObjId * nBisegments + bid;

            ob = op(bisegments, srcBid, dstBid, buffer, numElements * nBisegments);
        }
        offsetBytes += ob;
        return offsetBytes;
    }

#endif // __CUDACC__
};

/// \brief Helper class to construct a RodPackerHandler.
class RodPacker : public ObjectPacker
{
public:
    /** \brief Construct a RodPacker
        \param [in] predicate The channel filter that will be used to select the channels to be registered.
     */
    RodPacker(PackPredicate predicate);
    ~RodPacker();

    void update(LocalParticleVector *lpv, cudaStream_t stream) override;

    /// get a handler usable on device
    RodPackerHandler handler();

    size_t getSizeBytes(int numElements) const override;

protected:
    GenericPacker bisegmentData_;  ///< class that manages the bisegment data on host
    int nBisegments_; ///< number of bisegments per rod
};

} // namespace mirheo
