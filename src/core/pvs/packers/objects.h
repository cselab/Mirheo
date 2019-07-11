#pragma once

#include "particles.h"

class LocalObjectVector;

struct ObjectPackerHandler : public ParticlePackerHandler
{
    int objSize;
    GenericPackerHandler objects;

    inline __D__ size_t getSizeBytes(int numElements) const
    {
        return ParticlePackerHandler::getSizeBytes(numElements * objSize) +
            objects.getSizeBytes(numElements);
    }

    // utilities to pack / unpack one object per cuda block
    // used in exchangers
    
#ifdef __CUDACC__
    inline __device__ size_t blockPack(int numElements, char *buffer,
                                       int srcObjId, int dstObjId) const
    {
        return blockApply<PackOp>({}, numElements, buffer, srcObjId, dstObjId);
    }

    inline __device__ size_t blockPackShift(int numElements, char *buffer,
                                            int srcObjId, int dstObjId, float3 shift) const
    {
        return blockApply<PackShiftOp>({shift}, numElements, buffer, srcObjId, dstObjId);
    }

    inline __device__ size_t blockUnpack(int numElements, const char *buffer,
                                         int srcObjId, int dstObjId) const
    {
        return blockApply<UnpackOp>({}, numElements, buffer, srcObjId, dstObjId);
    }

    inline __device__ size_t blockUnpackAddNonZero(int numElements, const char *buffer,
                                                   int srcObjId, int dstObjId, float eps) const
    {
         return blockApply<UnpackAddOp>({eps}, numElements, buffer, srcObjId, dstObjId);
    }


protected:

    struct PackOp
    {
        inline __device__ auto operator()(const GenericPackerHandler& gpacker,
                                          int srcId, int dstId, char *buffer, int numElements)
        {
            return gpacker.pack(srcId, dstId, buffer, numElements);
        }
    };

    struct PackShiftOp
    {
        float3 shift;
        inline __device__ auto operator()(const GenericPackerHandler& gpacker,
                                          int srcId, int dstId, char *buffer, int numElements)
        {
            return gpacker.packShift(srcId, dstId, buffer, numElements, shift);
        }
    };

    struct UnpackOp
    {
        inline __device__ auto operator()(const GenericPackerHandler& gpacker,
                                          int srcId, int dstId, const char *buffer,
                                          int numElements)
        {
            return gpacker.unpack(srcId, dstId, buffer, numElements);
        }
    };

    struct UnpackAddOp
    {
        float eps;
        inline __device__ auto operator()(const GenericPackerHandler& gpacker,
                                          int srcId, int dstId, const char *buffer,
                                          int numElements)
        {
            return gpacker.unpackAtomicAddNonZero(srcId, dstId, buffer, numElements, eps);
        }
    };


    template <class Operation, typename BuffType>
    inline __device__ size_t blockApply(Operation op, int numElements, BuffType buffer,
                                        int srcObjId, int dstObjId) const
    {
        const int tid = threadIdx.x;
        __shared__ size_t offsetBytes;
        
        for (int pid = tid; pid < objSize; pid += blockDim.x)
        {
            const int srcPid = srcObjId * objSize + pid;
            const int dstPid = dstObjId * objSize + pid;
            
            size_t ob = op(particles, srcPid, dstPid, buffer, numElements * objSize);

            if (tid == 0)
                offsetBytes = ob;
        }
        __syncthreads();
        buffer += offsetBytes;
        
        if (tid == 0)
            offsetBytes += op(objects, srcObjId, dstObjId, buffer, numElements);

        __syncthreads();
        return offsetBytes;
    }
#endif // __CUDACC__
};

class ObjectPacker : public ParticlePacker
{
public:
    ObjectPacker(PackPredicate predicate);
    ~ObjectPacker();
    
    void update(LocalParticleVector *lpv, cudaStream_t stream) override;
    ObjectPackerHandler handler();
    size_t getSizeBytes(int numElements) const override;

protected:
    int objSize;
    GenericPacker objectData;
};
