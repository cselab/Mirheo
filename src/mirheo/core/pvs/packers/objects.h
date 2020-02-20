#pragma once

#include "particles.h"

namespace mirheo
{

class LocalObjectVector;

/** \brief A packer specific to objects.
    Will store both particle and object data into a single buffer.
 */
struct ObjectPackerHandler : public ParticlePackerHandler
{
    int objSize; ///< number of particles per object
    GenericPackerHandler objects; ///< packer responsible for the object data

    /// Get the reuired size (in bytes) of the buffer to hold the packed data
    __D__ size_t getSizeBytes(int numElements) const
    {
        return ParticlePackerHandler::getSizeBytes(numElements * objSize) +
            objects.getSizeBytes(numElements);
    }
    
#ifdef __CUDACC__

    /** \brief Fetch a full object from the registered channels and pack it into the buffer. 
        \param [in] numElements Number of objects that will be packed in the buffer.
        \param [out] buffer Destination buffer that will hold the packed object
        \param [in] srcObjId The index of the object to fetch from registered channels
        \param [in] dstObjId The index of the object to store into the buffer
        \return The size (in bytes) taken by the packed data (numElements objects). Only relevant for thread with Id 0.
        
        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockPack(int numElements, char *buffer,
                                int srcObjId, int dstObjId) const
    {
        return _blockApply<PackOp>({}, numElements, buffer, srcObjId, dstObjId);
    }

    /** \brief Fetch a full object from the registered channels, shift it and pack it into the buffer. 
        \param [in] numElements Number of objects that will be packed in the buffer.
        \param [out] buffer Destination buffer that will hold the packed object
        \param [in] srcObjId The index of the object to fetch from registered channels
        \param [in] dstObjId The index of the object to store into the buffer
        \param [in] shift The coordnate shift
        \return The size (in bytes) taken by the packed data (numElements objects). Only relevant for thread with Id 0.
        
        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockPackShift(int numElements, char *buffer,
                                     int srcObjId, int dstObjId, real3 shift) const
    {
        return _blockApply<PackShiftOp>({shift}, numElements, buffer, srcObjId, dstObjId);
    }

    /** \brief Unpack a full object from the buffer and store it into the registered channels. 
        \param [in] numElements Number of objects that will be packed in the buffer.
        \param [out] buffer Buffer that holds the packed object
        \param [in] srcObjId The index of the object to fetch from the buffer
        \param [in] dstObjId The index of the object to store into the registered channels
        \return The size (in bytes) taken by the packed data (numElements objects). Only relevant for thread with Id 0.
        
        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockUnpack(int numElements, const char *buffer,
                                  int srcObjId, int dstObjId) const
    {
        return _blockApply<UnpackOp>({}, numElements, buffer, srcObjId, dstObjId);
    }

    /** \brief Unpack a full object from the buffer and add it to the registered channels. 
        \param [in] numElements Number of objects that will be packed in the buffer.
        \param [out] buffer Buffer that holds the packed object
        \param [in] srcObjId The index of the object to fetch from the buffer
        \param [in] dstObjId The index of the object to store into the registered channels
        \param [in] eps Threshold under which the data will not be added
        \return The size (in bytes) taken by the packed data (numElements objects). Only relevant for thread with Id 0.
        
        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockUnpackAddNonZero(int numElements, const char *buffer,
                                            int srcObjId, int dstObjId, real eps) const
    {
         return _blockApply<UnpackAddOp>({eps}, numElements, buffer, srcObjId, dstObjId);
    }

    /** \brief Unpack a full object from the buffer, shift it and store it into the registered channels. 
        \param [in] numElements Number of objects that will be packed in the buffer.
        \param [out] buffer Buffer that holds the packed object
        \param [in] srcObjId The index of the object to fetch from the buffer
        \param [in] dstObjId The index of the object to store into the registered channels
        \param [in] shift Coordinates shift
        \return The size (in bytes) taken by the packed data (numElements objects). Only relevant for thread with Id 0.
        
        This method must be called by one CUDA block per object.
     */
    __device__ size_t blockUnpackShift(int numElements, const char *buffer,
                                       int srcObjId, int dstObjId, real3 shift) const
    {
        return _blockApply<UnpackShiftOp>({shift}, numElements, buffer, srcObjId, dstObjId);
    }

    /** \brief Copy the particle data of a full object from registered channels into the 
               registered channels of a ParticlePackerHandler
        \param [out] dst The destination ParticlePackerHandler
        \param [in] srcObjId The index of the object to fetch from the registered channels
        \param [in] dstPartIdOffset The index of the first particle in the destination ParticlePackerHandler
        
        This method must be called by one CUDA block per object.
     */
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

    /** \brief Copy a full object from registered channels into the registered channels of a ObjectPackerHandler
        \param [out] dst The destination ObjectPackerHandler
        \param [in] srcObjId The index of the object to fetch from the registered channels
        \param [in] dstObjId The index of the object to store in the destination ObjectPackerHandler
        
        This method must be called by one CUDA block per object.
     */
    __device__ void blockCopyTo(ObjectPackerHandler &dst, int srcObjId, int dstObjId) const
    {
        blockCopyParticlesTo(static_cast<ParticlePackerHandler &>(dst), srcObjId, dstObjId * objSize);

        const int tid = threadIdx.x;
        if (tid == 0)
            objects.copyTo(dst.objects, srcObjId, dstObjId);
    }

protected:
    /// Functor to pack data
    struct PackOp
    {
        /// apply the operation
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, char *buffer, int numElements)
        {
            return gpacker.pack(srcId, dstId, buffer, numElements);
        }
    };

    /// Functor to shift and pack data
    struct PackShiftOp
    {
        real3 shift; ///< coordinate shift
        /// apply the operation
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, char *buffer, int numElements)
        {
            return gpacker.packShift(srcId, dstId, buffer, numElements, shift);
        }
    };

    /// Functor to unpack and store data
    struct UnpackOp
    {
        /// apply the operation
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, const char *buffer,
                                   int numElements)
        {
            return gpacker.unpack(srcId, dstId, buffer, numElements);
        }
    };

    /// Functor to unpack and add data atomically
    struct UnpackAddOp
    {
        real eps; ///< tolerance to add data. Data won't be added if is smaller than this.
        /// apply the operation
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, const char *buffer,
                                   int numElements)
        {
            return gpacker.unpackAtomicAddNonZero(srcId, dstId, buffer, numElements, eps);
        }
    };

    /// Functor to unpack, shift and store data
    struct UnpackShiftOp
    {
        real3 shift;  ///< coordinate shift
        /// apply the operation
        __device__ auto operator()(const GenericPackerHandler& gpacker,
                                   int srcId, int dstId, const char *buffer,
                                   int numElements)
        {
            return gpacker.unpackShift(srcId, dstId, buffer, numElements, shift);
        }
    };

    /** \brief Apply an operation on all data of a single object.
        \tparam Operation The operation to apply to the datum
        \tparam BuffType Must be const char* or char* depending on the constness required by the operation
        \param [in] op The operation functor
        \param [in] numElements Number of objects to pack
        \param buffer Source or destination buffer, depending on op
        \param [in] srcObjId The index of the object to fetch
        \param [in] dstObjId The index of the object to store
        \return The size (in bytes) taken by the packed data (numElements objects). Only relevant for thread with Id 0.

        This method expects to be executed by a full CUDA block per object.
     */
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

/// \brief Helper class to construct a ObjectPackerHandler.
class ObjectPacker : public ParticlePacker
{
public:
    /** \brief Construct a ObjectPacker
        \param [in] predicate The channel filter that will be used to select the channels to be registered.
     */
    ObjectPacker(PackPredicate predicate);
    ~ObjectPacker();
    
    void update(LocalParticleVector *lpv, cudaStream_t stream) override;

    /// get a handler usable on device
    ObjectPackerHandler handler();

    size_t getSizeBytes(int numElements) const override;

protected:
    int objSize_; ///< number of particles per object
    GenericPacker objectData_; ///< class that manage the object data on host
};

} // namespace mirheo
