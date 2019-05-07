#include "objects.h"
#include "common.h"
#include "shifter.h"
#include "../exchange_helpers.h"

#include <core/pvs/object_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <type_traits>

namespace ObjectPackerKernels
{
template <typename T>
__global__ void packParticlesToBuffer(const MapEntry *map, int objSize, const size_t *offsetsBytes,
                                      const int *offsets, const T *srcData, Shifter shift, char *buffer)
{
    int objId = blockIdx.x;
    auto m = map[objId];
    int buffId = m.getBufId();
    int  srcId = m.getId();
    T *dstData = (T*) (buffer + offsetsBytes[buffId]);
    int bufOffset = offsets[buffId];

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int dstId = (objId - bufOffset) * objSize + i;
        dstData[dstId] = shift(srcData[srcId + i], buffId);
    }
}

template <typename T>
__global__ void packObjectsToBuffer(int nObj, const MapEntry *map, const size_t *offsetsBytes,
                                    const int *offsets, const T *srcData, Shifter shift, char *buffer)
{
    int objId = threadIdx.x + blockIdx.x * blockDim.x;

    if (objId >= nObj) return;
    
    auto m = map[objId];
    int buffId = m.getBufId();
    int  srcId = m.getId();
    T *dstData = (T*) (buffer + offsetsBytes[buffId]);
    int bufOffset = offsets[buffId];

    int dstId = objId - bufOffset;
    dstData[dstId] = shift(srcData[srcId], buffId);
}

template <typename T>
__global__ void unpackParticlesFromBuffer(int nBuffers, const int *offsets, int objSize, const char *buffer, const size_t *offsetsBytes, T *dstData)
{
    int objId = blockIdx.x;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();

    int buffId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    objId -= sharedOffsets[buffId];
    
    const T *srcData = (const T*) (buffer + offsetsBytes[buffId]);

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int j = objId * objSize + i;
        dstData[j] = srcData[j];
    }
}

template <typename T>
__global__ void unpackObjectsFromBuffer(int nObj, int nBuffers, const int *offsets, const char *buffer, const size_t *offsetsBytes, T *dstData)
{
    int objId = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();

    if (objId >= nObj) return;
    
    int buffId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    objId -= sharedOffsets[buffId];
    
    const T *srcData = (const T*) (buffer + offsetsBytes[buffId]);

    dstData[objId] = srcData[objId];
}

} // namespace ObjectPackerKernels

ObjectsPacker::ObjectsPacker(ObjectVector *ov, PackPredicate predicate) :
    Packer(ov, predicate),
    ov(ov)
{}

size_t ObjectsPacker::getPackedSizeBytes(int nobj) const
{
    auto packedSizeParts = _getPackedSizeBytes(ov->local()->dataPerParticle, nobj * ov->objSize);
    auto packedSizeObjs  = _getPackedSizeBytes(ov->local()->dataPerObject,   nobj);

    return packedSizeParts + packedSizeObjs;
}

void ObjectsPacker::packToBuffer(const LocalObjectVector *lov, ExchangeHelper *helper, cudaStream_t stream)
{
    auto& partManager = lov->dataPerParticle;
    auto& objManager  = lov->dataPerObject;

    offsetsBytes.copyFromDevice(helper->send.offsetsBytes, stream);

    // pack particle data
    for (const auto& name_desc : partManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        Shifter shift(desc->shiftTypeSize > 0, ov->state->domain);
        
        auto packChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nObj = helper->map.size();
            const int nthreads = 128;
            
            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::packParticlesToBuffer,
                nObj, nthreads, 0, stream,
                helper->map.devPtr(), ov->objSize, offsetsBytes.devPtr(), helper->send.offsets.devPtr(),
                pinnedBuffPtr->devPtr(), shift, helper->send.buffer.devPtr());

            updateOffsetsObjects<T>(helper->nBuffers, ov->objSize, helper->send.sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(packChannel, desc->varDataPtr);
    }

    // pack object data
    for (const auto& name_desc : objManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        Shifter shift(desc->shiftTypeSize > 0, ov->state->domain);
        
        auto packChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nObj = helper->map.size();
            const int nthreads = 32;

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::packObjectsToBuffer,
                getNblocks(nObj, nthreads), nthreads, 0, stream,
                nObj, helper->map.devPtr(), offsetsBytes.devPtr(), helper->send.offsets.devPtr(),
                pinnedBuffPtr->devPtr(), shift, helper->send.buffer.devPtr());

            updateOffsets<T>(helper->nBuffers, helper->send.sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(packChannel, desc->varDataPtr);
    }
}

void ObjectsPacker::unpackFromBuffer(LocalObjectVector *lov, const BufferInfos *helper, int oldObjSize, cudaStream_t stream)
{
    auto& partManager = lov->dataPerParticle;
    auto& objManager  = lov->dataPerObject;

    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    // unpack particle data
    for (const auto& name_desc : partManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto unpackChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            int nBuffers = helper->sizes.size();
            int nObj     = helper->offsets[nBuffers];
            const int nthreads = 128;
            const size_t sharedMem = nBuffers * sizeof(int);

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::unpackParticlesFromBuffer,
                nObj, nthreads, sharedMem, stream,
                nBuffers, helper->offsets.devPtr(), ov->objSize,
                helper->buffer.devPtr(), offsetsBytes.devPtr(),
                pinnedBuffPtr->devPtr() + oldObjSize * ov->objSize);

            updateOffsetsObjects<T>(nBuffers, ov->objSize, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(unpackChannel, desc->varDataPtr);
    }

    // unpack object data
    for (const auto& name_desc : objManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto unpackChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            int nBuffers = helper->sizes.size();
            int nObj     = helper->offsets[nBuffers];
            const int nthreads = 32;
            const size_t sharedMem = nBuffers * sizeof(int);

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::unpackObjectsFromBuffer,
                getNblocks(nObj, nthreads), nthreads, sharedMem, stream,
                nObj, nBuffers, helper->offsets.devPtr(), helper->buffer.devPtr(),
                offsetsBytes.devPtr(), pinnedBuffPtr->devPtr() + oldObjSize);

            updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(unpackChannel, desc->varDataPtr);
    }
}
