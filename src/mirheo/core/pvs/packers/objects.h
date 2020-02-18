#pragma once

#include "particles.h"

namespace mirheo
{

class LocalObjectVector;

struct ObjectPackerHandler : public ParticlePackerHandler
{
    int objSize;
    GenericPackerHandler objects;

    __D__ size_t getSizeBytes(int numElements) const
    {
        return ParticlePackerHandler::getSizeBytes(numElements * objSize) +
            objects.getSizeBytes(numElements);
    }

    // utilities to pack / unpack one object per cuda block
    // used in exchangers
    
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

    __device__ size_t blockUnpackShift(int numElements, const char *buffer,
                                       int srcObjId, int dstObjId, real3 shift) const
    {
        return _blockApply<UnpackShiftOp>({shift}, numElements, buffer, srcObjId, dstObjId);
    }

    __device__ void blockCopyParticlesTo(ParticlePackerHandler &dst, int srcObjId, int dstPartIdOffset) const
    {
        const int tid = threadIdx.x;
        for (int pid = tid; pid < objSize; pid += blockDim.x)
        {
            const int srcPid = srcObjId * objSize + pid;
            const int dstPid = dstPartIdOffset + pid;

            particles.copyTo(dst.particles, srcPid, dstPid);
        }
    }

    __device__ void blockCopyTo(ObjectPackerHandler &dst, int srcObjId, int dstObjId) const
    {
        blockCopyParticlesTo(static_cast<ParticlePackerHandler &>(dst), srcObjId, dstObjId * objSize);

        const int tid = threadIdx.x;
        if (tid == 0)
            objects.copyTo(dst.objects, srcObjId, dstObjId);
    }

protected:
    struct PackOp
    {
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, char *buffer, int numElements)
        {
            return gpacker.pack(srcId, dstId, buffer, numElements);
        }
    };

    struct PackShiftOp
    {
        real3 shift;
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, char *buffer, int numElements)
        {
            return gpacker.packShift(srcId, dstId, buffer, numElements, shift);
        }
    };

    struct UnpackOp
    {
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, const char *buffer,
                                   int numElements)
        {
            return gpacker.unpack(srcId, dstId, buffer, numElements);
        }
    };

    struct UnpackAddOp
    {
        real eps;
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, const char *buffer,
                                   int numElements)
        {
            return gpacker.unpackAtomicAddNonZero(srcId, dstId, buffer, numElements, eps);
        }
    };

    struct UnpackShiftOp
    {
        real3 shift;
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, const char *buffer,
                                   int numElements)
        {
            return gpacker.unpackShift(srcId, dstId, buffer, numElements, shift);
        }
    };


    template <class Operation, typename BuffType>
    __device__ size_t _blockApply(Operation op, int numElements, BuffType buffer,
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
    int objSize_;
    GenericPacker objectData_;
};

} // namespace mirheo
