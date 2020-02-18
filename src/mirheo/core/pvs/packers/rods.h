#pragma once

#include "objects.h"

namespace mirheo
{

class LocalRodVector;

struct RodPackerHandler : public ObjectPackerHandler
{
    int nBisegments;
    GenericPackerHandler bisegments;

    __D__ size_t getSizeBytes(int numElements) const
    {
        return ObjectPackerHandler::getSizeBytes(numElements) +
            bisegments.getSizeBytes(numElements * nBisegments);
    }

#ifdef __CUDACC__
    __device__ size_t blockPack(int numElements, char *buffer,
                                int srcObjId, int dstObjId) const
    {
        return _blockApply<PackOp>({}, numElements, buffer, srcObjId, dstObjId);
    }

    __device__ size_t blockPackShift(int numElements, char *buffer,
                                     int srcObjId, int dstObjId, real3 shift) const
    {
        return _blockApply<PackShiftOp>({shift}, numElements, buffer, srcObjId, dstObjId);
    }

    __device__ size_t blockUnpack(int numElements, const char *buffer,
                                  int srcObjId, int dstObjId) const
    {
        return _blockApply<UnpackOp>({}, numElements, buffer, srcObjId, dstObjId);
    }

    __device__ size_t blockUnpackAddNonZero(int numElements, const char *buffer,
                                            int srcObjId, int dstObjId, real eps) const
    {
         return _blockApply<UnpackAddOp>({eps}, numElements, buffer, srcObjId, dstObjId);
    }

protected:

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


class RodPacker : public ObjectPacker
{
public:

    RodPacker(PackPredicate predicate);
    ~RodPacker();
    
    void update(LocalParticleVector *lpv, cudaStream_t stream) override;
    RodPackerHandler handler();
    size_t getSizeBytes(int numElements) const override;

protected:
    GenericPacker bisegmentData;
    int nBisegments;
};

} // namespace mirheo
