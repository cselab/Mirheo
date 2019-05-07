#include "objects.h"
#include "common.h"
#include "adder.h"
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
    int bufId = m.getBufId();
    int  srcId = m.getId();
    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[bufId]);
    int bufOffset = offsets[bufId];

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int dstId = (objId - bufOffset) * objSize + i;
        dstData[dstId] = shift(srcData[srcId + i], bufId);
    }
}

template <typename T>
__global__ void packObjectsToBuffer(int nObj, const MapEntry *map, const size_t *offsetsBytes,
                                    const int *offsets, const T *srcData, Shifter shift, char *buffer)
{
    int objId = threadIdx.x + blockIdx.x * blockDim.x;

    if (objId >= nObj) return;
    
    auto m = map[objId];
    int bufId = m.getBufId();
    int  srcId = m.getId();
    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[bufId]);
    int bufOffset = offsets[bufId];

    int dstId = objId - bufOffset;
    dstData[dstId] = shift(srcData[srcId], bufId);
}

template <typename T>
__global__ void unpackParticlesFromBuffer(int nBuffers, const int *offsets, int objSize, const char *buffer, const size_t *offsetsBytes, T *dstData)
{
    int objId = blockIdx.x;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();

    int bufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    objId -= sharedOffsets[bufId];
    
    auto srcData = reinterpret_cast<const T*>(buffer + offsetsBytes[bufId]);

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
    
    int bufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    objId -= sharedOffsets[bufId];
    
    auto srcData = reinterpret_cast<const T*>(buffer + offsetsBytes[bufId]);

    dstData[objId] = srcData[objId];
}



template <typename T>
__global__ void reversePackParticlesToBuffer(int nBuffers, const int *offsets, int objSize, const T *srcData, const size_t *offsetsBytes, char *buffer)
{
    int objId = blockIdx.x;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();

    int bufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    objId -= sharedOffsets[bufId];
    
    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[bufId]);

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int j = objId * objSize + i;
        dstData[j] = srcData[j];
    }
}

template <typename T>
__global__ void reversePackObjectsToBuffer(int nObj, int nBuffers, const T *srcData, const int *offsets, const size_t *offsetsBytes, char *buffer)
{
    int objId = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();

    if (objId >= nObj) return;
    
    int bufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    objId -= sharedOffsets[bufId];
    
    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[bufId]);

    dstData[objId] = srcData[objId];
}


template <typename T>
__global__ void reverseUnpackAndAddParticlesToBuffer(const MapEntry *map, int objSize, const size_t *offsetsBytes,
                                                     const int *offsets, const char *buffer, AtomicAdder add, T *dstData)
{
    int objId = blockIdx.x;
    auto m = map[objId];
    int bufId = m.getBufId();
    int  dstId = m.getId();
    auto srcData = reinterpret_cast<const T*>(buffer + offsetsBytes[bufId]);
    int bufOffset = offsets[bufId];

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int srcId = (objId - bufOffset) * objSize + i;
        add(dstData + dstId + i, srcData[srcId]);
    }
}

template <typename T>
__global__ void reverseUnpackAndAddObjectsToBuffer(int nObj, const MapEntry *map, const size_t *offsetsBytes,
                                                   const int *offsets, const char *buffer, AtomicAdder add, T *dstData)
{
    int objId = threadIdx.x + blockIdx.x * blockDim.x;

    if (objId >= nObj) return;
    
    auto m = map[objId];
    int bufId = m.getBufId();
    int  dstId = m.getId();
    auto srcData = reinterpret_cast<const T*>(buffer + offsetsBytes[bufId]);
    int bufOffset = offsets[bufId];

    int srcId = objId - bufOffset;
    add(dstData + dstId, srcData[srcId]);
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

void ObjectsPacker::packToBuffer(const LocalObjectVector *lov, DeviceBuffer<MapEntry>& map, BufferInfos *helper, cudaStream_t stream)
{
    auto& partManager = lov->dataPerParticle;
    auto& objManager  = lov->dataPerObject;

    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    int nBuffers = helper->sizes.size();
    
    // pack particle data
    for (const auto& name_desc : partManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        Shifter shift(desc->shiftTypeSize > 0, ov->state->domain);
        
        auto packChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nObj = map.size();
            const int nthreads = 128;
            
            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::packParticlesToBuffer,
                nObj, nthreads, 0, stream,
                map.devPtr(), ov->objSize, offsetsBytes.devPtr(), helper->offsets.devPtr(),
                pinnedBuffPtr->devPtr(), shift, helper->buffer.devPtr());

            updateOffsetsObjects<T>(nBuffers, ov->objSize, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
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

            const int nObj = map.size();
            const int nthreads = 32;

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::packObjectsToBuffer,
                getNblocks(nObj, nthreads), nthreads, 0, stream,
                nObj, map.devPtr(), offsetsBytes.devPtr(), helper->offsets.devPtr(),
                pinnedBuffPtr->devPtr(), shift, helper->buffer.devPtr());

            updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
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

void ObjectsPacker::reversePackToBuffer(const LocalObjectVector *lov, BufferInfos *helper, cudaStream_t stream)
{
    auto& partManager = lov->dataPerParticle;
    auto& objManager  = lov->dataPerObject;

    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    // reverse pack particle data
    for (const auto& name_desc : partManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto reversePackChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            int nBuffers = helper->sizes.size();
            int nObj     = helper->offsets[nBuffers];
            const int nthreads = 128;
            const size_t sharedMem = nBuffers * sizeof(int);

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::reversePackParticlesToBuffer,
                nObj, nthreads, sharedMem, stream,
                nBuffers, helper->offsets.devPtr(), ov->objSize,
                pinnedBuffPtr->devPtr(), offsetsBytes.devPtr(),
                helper->buffer.devPtr());

            updateOffsetsObjects<T>(nBuffers, ov->objSize, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(reversePackChannel, desc->varDataPtr);
    }

    // reverse pack object data
    for (const auto& name_desc : objManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto reversePackChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            int nBuffers = helper->sizes.size();
            int nObj     = helper->offsets[nBuffers];
            const int nthreads = 32;
            const size_t sharedMem = nBuffers * sizeof(int);

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::reversePackObjectsToBuffer,
                getNblocks(nObj, nthreads), nthreads, sharedMem, stream,
                nObj, nBuffers, pinnedBuffPtr->devPtr(), helper->offsets.devPtr(),
                offsetsBytes.devPtr(), helper->buffer.devPtr());

            updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(reversePackChannel, desc->varDataPtr);
    }

}

void ObjectsPacker::reverseUnpackFromBufferAndAdd(LocalObjectVector *lov, const DeviceBuffer<MapEntry>& map,
                                                  const BufferInfos *helper, cudaStream_t stream)
{
    auto& partManager = lov->dataPerParticle;
    auto& objManager  = lov->dataPerObject;

    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    int nBuffers = helper->sizes.size();
    AtomicAdder adder(1e-6f);
    
    // pack particle data
    for (const auto& name_desc : partManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        Shifter shift(desc->shiftTypeSize > 0, ov->state->domain);
        
        auto reverseUnpackAndAddChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nObj = map.size();
            const int nthreads = 128;
            
            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::reverseUnpackAndAddParticlesToBuffer,
                nObj, nthreads, 0, stream,
                map.devPtr(), ov->objSize, offsetsBytes.devPtr(), helper->offsets.devPtr(),
                helper->buffer.devPtr(), adder, pinnedBuffPtr->devPtr());

            updateOffsetsObjects<T>(nBuffers, ov->objSize, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(reverseUnpackAndAddChannel, desc->varDataPtr);
    }

    // pack object data
    for (const auto& name_desc : objManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        Shifter shift(desc->shiftTypeSize > 0, ov->state->domain);
        
        auto reverseUnpackAndAddChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nObj = map.size();
            const int nthreads = 32;

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::reverseUnpackAndAddObjectsToBuffer,
                getNblocks(nObj, nthreads), nthreads, 0, stream,
                nObj, map.devPtr(), offsetsBytes.devPtr(), helper->offsets.devPtr(),
                helper->buffer.devPtr(), adder, pinnedBuffPtr->devPtr());

            updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(reverseUnpackAndAddChannel, desc->varDataPtr);
    }    
}
