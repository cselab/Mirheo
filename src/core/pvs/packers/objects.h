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
    inline __device__ size_t blockPackShift(int numElements, char *buffer,
                                            int srcObjId, int dstObjId, float3 shift) const
    {
        const int tid = threadIdx.x;
        __shared__ size_t offsetBytes;
        
        for (int pid = tid; pid < objSize; pid += blockDim.x)
        {
            const int srcPid = srcObjId * objSize + pid;
            const int dstPid = dstObjId * objSize + pid;
            
            size_t ob = particles.packShift(srcPid, dstPid, buffer,
                                            numElements * objSize, shift);

            if (tid == 0)
                offsetBytes = ob;
        }
        __syncthreads();
        buffer += offsetBytes;
        
        if (tid == 0)
            offsetBytes += objects.packShift(srcObjId, dstObjId,
                                             buffer, numElements, shift);

        __syncthreads();
        return offsetBytes;
    }

    inline __device__ size_t blockUnpack(int numElements, const char *buffer,
                                         int srcObjId, int dstObjId) const
    {
        const int tid = threadIdx.x;
        __shared__ size_t offsetBytes;
        
        for (int pid = tid; pid < objSize; pid += blockDim.x)
        {
            const int srcPid = srcObjId * objSize + pid;
            const int dstPid = dstObjId * objSize + pid;
            
            size_t ob = particles.unpack(srcPid, dstPid, buffer,
                                         numElements * objSize);

            if (tid == 0)
                offsetBytes = ob;
        }
        __syncthreads();
        buffer += offsetBytes;
        
        if (tid == 0)
            offsetBytes += objects.unpack(srcObjId, dstObjId,
                                          buffer, numElements);

        __syncthreads();
        return offsetBytes;
    }
#endif
};

class ObjectPacker : public ParticlePacker
{
public:
    ObjectPacker(PackPredicate predicate);
    ~ObjectPacker();
    
    void update(LocalObjectVector *lov, cudaStream_t stream);
    ObjectPackerHandler handler();
    size_t getSizeBytes(int numElements) const override;

protected:
    int objSize;
    GenericPacker objectData;
};
