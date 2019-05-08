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
    int dstBufId = m.getBufId();
    int srcObjId = m.getId();

    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[dstBufId]);
    int dstObjId = objId - offsets[dstBufId];

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int dstId = dstObjId * objSize + i;
        int srcId = srcObjId * objSize + i;

        dstData[dstId] = shift(srcData[srcId], dstBufId);
    }
}

template <typename T>
__global__ void packObjectsToBuffer(int nObj, const MapEntry *map, const size_t *offsetsBytes,
                                    const int *offsets, const T *srcData, Shifter shift, char *buffer)
{
    int objId = threadIdx.x + blockIdx.x * blockDim.x;

    if (objId >= nObj) return;
    
    auto m = map[objId];
    int dstBufId = m.getBufId();
    int srcObjId = m.getId();

    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[dstBufId]);
    int dstObjId = objId - offsets[dstBufId];

    dstData[dstObjId] = shift(srcData[srcObjId], dstBufId);
}

template <typename T>
__global__ void unpackParticlesFromBuffer(int nBuffers, const int *offsets, int objSize, const char *buffer, const size_t *offsetsBytes, T *dstData)
{
    int objId = blockIdx.x;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();

    int srcBufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    int srcObjId = objId + sharedOffsets[0] - sharedOffsets[srcBufId];
    
    auto srcData = reinterpret_cast<const T*>(buffer + offsetsBytes[srcBufId]);

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int srcId = srcObjId * objSize + i;
        int dstId =    objId * objSize + i;

        dstData[dstId] = srcData[srcId];
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

    int srcBufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    int srcObjId = objId + sharedOffsets[0] - sharedOffsets[srcBufId];

    auto srcData = reinterpret_cast<const T*>(buffer + offsetsBytes[srcBufId]);

    dstData[objId] = srcData[srcObjId];
}



template <typename T>
__global__ void reversePackParticlesToBuffer(int nBuffers, const int *offsets, int objSize, const T *srcData, const size_t *offsetsBytes, char *buffer)
{
    int objId = blockIdx.x;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();

    int dstBufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    int dstObjId = objId - sharedOffsets[dstBufId];
    
    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[dstBufId]);

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int srcId =    objId * objSize + i;
        int dstId = dstObjId * objSize + i;

        dstData[dstId] = srcData[srcId];
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
    
    int dstBufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, objId);
    int dstObjId = objId - sharedOffsets[dstBufId];
    
    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[dstBufId]);

    dstData[dstObjId] = srcData[objId];
}


template <typename T>
__global__ void reverseUnpackAndAddParticlesFromBuffer(const MapEntry *map, int objSize, const size_t *offsetsBytes,
                                                       const int *offsets, const char *buffer, AtomicAdder atomicAddNonZero, T *dstData)
{
    int objId = blockIdx.x;

    auto m = map[objId];
    int srcBufId = m.getBufId();
    int dstObjId = m.getId();
    
    auto srcData = reinterpret_cast<const T*>(buffer + offsetsBytes[srcBufId]);
    int srcObjId = objId - offsets[srcBufId];

    for (int i = threadIdx.x; i < objSize; i += blockDim.x)
    {
        int srcId = srcObjId * objSize + i;
        int dstId = dstObjId * objSize + i;

        atomicAddNonZero(&dstData[dstId], srcData[srcId]);
    }
}

template <typename T>
__global__ void reverseUnpackAndAddObjectsFromBuffer(int nObj, const MapEntry *map, const size_t *offsetsBytes,
                                                     const int *offsets, const char *buffer, AtomicAdder atomicAddNonZero, T *dstData)
{
    int objId = threadIdx.x + blockIdx.x * blockDim.x;

    if (objId >= nObj) return;
    
    auto m = map[objId];
    int srcBufId = m.getBufId();
    int dstObjId = m.getId();

    auto srcData = reinterpret_cast<const T*>(buffer + offsetsBytes[srcBufId]);
    int srcObjId = objId - offsets[srcBufId];

    atomicAddNonZero(&dstData[dstObjId], srcData[srcObjId]);
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

void ObjectsPacker::packToBuffer(const LocalObjectVector *lov, const DeviceBuffer<MapEntry>& map, BufferInfos *helper, cudaStream_t stream)
{
    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    int nBuffers = helper->sizes.size();
    
    auto packParts = [&](auto pinnedBuffPtr, const auto& nameDesc)
    {
        using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;
        auto& desc = nameDesc.second;

        bool needShift = desc->shiftTypeSize > 0;
        Shifter shift(needShift, ov->state->domain);

        const int nObj = map.size();
        const int nthreads = 128;

        debug2("Packing %d object particles '%s' of '%s' %s shift",
               nObj * ov->objSize, nameDesc.first.c_str(),
               ov->name.c_str(), needShift ? "with" : "with no");
        
        SAFE_KERNEL_LAUNCH(
            ObjectPackerKernels::packParticlesToBuffer,
            nObj, nthreads, 0, stream,
            map.devPtr(), ov->objSize, offsetsBytes.devPtr(), helper->offsets.devPtr(),
            pinnedBuffPtr->devPtr(), shift, helper->buffer.devPtr());

        updateOffsetsObjects<T>(nBuffers, ov->objSize, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
    };

    auto packObjs = [&](auto pinnedBuffPtr, const auto& nameDesc)
    {
        using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;
        auto& desc = nameDesc.second;

        bool needShift = desc->shiftTypeSize > 0;
        Shifter shift(needShift, ov->state->domain);

        const int nObj = map.size();
        const int nthreads = 32;

        debug2("Packing %d object quantities '%s' of '%s' %s shift",
               nObj, nameDesc.first.c_str(), ov->name.c_str(),
               needShift ? "with" : "with no");

        SAFE_KERNEL_LAUNCH(
            ObjectPackerKernels::packObjectsToBuffer,
            getNblocks(nObj, nthreads), nthreads, 0, stream,
            nObj, map.devPtr(), offsetsBytes.devPtr(), helper->offsets.devPtr(),
            pinnedBuffPtr->devPtr(), shift, helper->buffer.devPtr());

        updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
    };

    _applyToChannels(lov, packParts, packObjs);
}

void ObjectsPacker::unpackFromBuffer(LocalObjectVector *lov, const BufferInfos *helper, int oldObjSize, cudaStream_t stream)
{
    const int bufStart = 0;
    const int bufEnd   = helper->sizes.size();
    
    _unpackFromBuffer(lov, helper, oldObjSize, bufStart, bufEnd, stream);
}

void ObjectsPacker::unpackBulkFromBuffer(LocalObjectVector *lov, int bulkId, const BufferInfos *helper, cudaStream_t stream)
{
    const int bufStart   = bulkId;
    const int bufEnd     = bulkId + 1;
    const int oldObjSize = 0;
    
    _unpackFromBuffer(lov, helper, oldObjSize, bufStart, bufEnd, stream);
}

void ObjectsPacker::_unpackFromBuffer(LocalObjectVector *lov, const BufferInfos *helper, int oldObjSize, int bufStart, int bufEnd, cudaStream_t stream)
{
    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    int nBuffers = bufEnd - bufStart;
    int nObj     = helper->offsets[bufEnd] - helper->offsets[bufStart];

    auto sizesPtr        = helper->sizes  .devPtr() + bufStart;
    auto offsetsPtr      = helper->offsets.devPtr() + bufStart;
    auto offsetsBytesPtr = offsetsBytes   .devPtr() + bufStart;

    auto unpackParts = [&](auto pinnedBuffPtr, const auto& nameDesc)
    {
        using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

        debug2("Unpacking object particles '%s' of '%s'", nameDesc.first.c_str(), ov->name.c_str());
        
        const int nthreads = 128;
        const size_t sharedMem = nBuffers * sizeof(int);

        SAFE_KERNEL_LAUNCH(
            ObjectPackerKernels::unpackParticlesFromBuffer,
            nObj, nthreads, sharedMem, stream,
            nBuffers, offsetsPtr, ov->objSize,
            helper->buffer.devPtr(), offsetsBytesPtr,
            pinnedBuffPtr->devPtr() + oldObjSize * ov->objSize);

        updateOffsetsObjects<T>(nBuffers, ov->objSize, sizesPtr, offsetsBytesPtr, stream);
    };
        
    auto unpackObjs = [&](auto pinnedBuffPtr, const auto& nameDesc)
    {
        using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

        debug2("Unpacking object quantities '%s' of '%s'", nameDesc.first.c_str(), ov->name.c_str());
        
        const int nthreads = 32;
        const size_t sharedMem = nBuffers * sizeof(int);

        SAFE_KERNEL_LAUNCH(
            ObjectPackerKernels::unpackObjectsFromBuffer,
            getNblocks(nObj, nthreads), nthreads, sharedMem, stream,
            nObj, nBuffers, offsetsPtr, helper->buffer.devPtr(),
            offsetsBytesPtr, pinnedBuffPtr->devPtr() + oldObjSize);

        updateOffsets<T>(nBuffers, sizesPtr, offsetsBytesPtr, stream);
    };

    _applyToChannels(lov, unpackParts, unpackObjs);
}

void ObjectsPacker::reversePackToBuffer(const LocalObjectVector *lov, BufferInfos *helper, cudaStream_t stream)
{
    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    auto reversePackParts = [&](auto pinnedBuffPtr, const auto& nameDesc)
    {
        using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

        int nBuffers = helper->sizes.size();
        int nObj     = helper->offsets[nBuffers];
        const int nthreads = 128;
        const size_t sharedMem = nBuffers * sizeof(int);

        debug2("Reverse packing %d object particles '%s' of '%s'",
               nObj * ov->objSize, nameDesc.first.c_str(), ov->name.c_str());
        
        SAFE_KERNEL_LAUNCH(
            ObjectPackerKernels::reversePackParticlesToBuffer,
            nObj, nthreads, sharedMem, stream,
            nBuffers, helper->offsets.devPtr(), ov->objSize,
            pinnedBuffPtr->devPtr(), offsetsBytes.devPtr(),
            helper->buffer.devPtr());

        updateOffsetsObjects<T>(nBuffers, ov->objSize, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
    };

    auto reversePackObjs = [&](auto pinnedBuffPtr, const auto& nameDesc)
    {
        using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

        int nBuffers = helper->sizes.size();
        int nObj     = helper->offsets[nBuffers];
        const int nthreads = 32;
        const size_t sharedMem = nBuffers * sizeof(int);

        debug2("Reverse packing %d object quantities '%s' of '%s'",
               nObj, nameDesc.first.c_str(), ov->name.c_str());

        SAFE_KERNEL_LAUNCH(
            ObjectPackerKernels::reversePackObjectsToBuffer,
            getNblocks(nObj, nthreads), nthreads, sharedMem, stream,
            nObj, nBuffers, pinnedBuffPtr->devPtr(), helper->offsets.devPtr(),
            offsetsBytes.devPtr(), helper->buffer.devPtr());

        updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
    };

    _applyToChannels(lov, reversePackParts, reversePackObjs);
}

void ObjectsPacker::reverseUnpackFromBufferAndAdd(LocalObjectVector *lov, const DeviceBuffer<MapEntry>& map,
                                                  const BufferInfos *helper, cudaStream_t stream)
{
    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    int nBuffers = helper->sizes.size();
    AtomicAdder adder(1e-6f);
        
    auto reverseUnpackAndAddParts = [&](auto pinnedBuffPtr, const auto& nameDesc)
    {
        using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

        auto& desc = nameDesc.second;
        Shifter shift(desc->shiftTypeSize > 0, ov->state->domain);
        
        const int nObj = map.size();
        const int nthreads = 128;

        debug2("Reverse unpacking %d object particles '%s' of '%s'",
               nObj * ov->objSize, nameDesc.first.c_str(), ov->name.c_str());
        
        SAFE_KERNEL_LAUNCH(
            ObjectPackerKernels::reverseUnpackAndAddParticlesFromBuffer,
            nObj, nthreads, 0, stream,
            map.devPtr(), ov->objSize, offsetsBytes.devPtr(), helper->offsets.devPtr(),
            helper->buffer.devPtr(), adder, pinnedBuffPtr->devPtr());

        updateOffsetsObjects<T>(nBuffers, ov->objSize, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
    };
        
    auto reverseUnpackAndAddObjs = [&](auto pinnedBuffPtr, const auto& nameDesc)
    {
        using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

        auto& desc = nameDesc.second;
        Shifter shift(desc->shiftTypeSize > 0, ov->state->domain);

        const int nObj = map.size();
        const int nthreads = 32;

        debug2("Reverse unpacking %d object entities '%s' of '%s'",
               nObj, nameDesc.first.c_str(), ov->name.c_str());

        SAFE_KERNEL_LAUNCH(
            ObjectPackerKernels::reverseUnpackAndAddObjectsFromBuffer,
            getNblocks(nObj, nthreads), nthreads, 0, stream,
            nObj, map.devPtr(), offsetsBytes.devPtr(), helper->offsets.devPtr(),
            helper->buffer.devPtr(), adder, pinnedBuffPtr->devPtr());

        updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
    };

    _applyToChannels(lov, reverseUnpackAndAddParts, reverseUnpackAndAddObjs);
}

template <typename Pvisitor, typename Ovisitor>
void ObjectsPacker::_applyToChannels(const LocalObjectVector *lov, Pvisitor &&pvisitor, Ovisitor &&ovisitor)
{
    auto& partManager = lov->dataPerParticle;
    auto& objManager  = lov->dataPerObject;

    // visit particle data
    for (const auto& nameDesc : partManager.getSortedChannels())
    {
        if (!predicate(nameDesc)) continue;
        auto& desc = nameDesc.second;

        auto visitor = [&](auto pinnedBuffPtr) {return pvisitor(pinnedBuffPtr, nameDesc);};
        mpark::visit(visitor, desc->varDataPtr);
    }

    // visit object data
    for (const auto& nameDesc : objManager.getSortedChannels())
    {
        if (!predicate(nameDesc)) continue;
        auto& desc = nameDesc.second;

        auto visitor = [&](auto pinnedBuffPtr) {return ovisitor(pinnedBuffPtr, nameDesc);};
        mpark::visit(visitor, desc->varDataPtr);
    }
}

