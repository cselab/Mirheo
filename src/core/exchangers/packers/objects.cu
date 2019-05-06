#include "objects.h"
#include "common.h"

#include <core/pvs/object_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <type_traits>

namespace ObjectPackerKernels
{
template <typename T>
__global__ void packParticlesToBuffer(const MapEntry *map, int objSize, const size_t *offsetsBytes,
                                      const int *offsets, const T *srcData, char *buffer)
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
        dstData[dstId] = srcData[srcId + i]; // TODO shift
    }
}

template <typename T>
__global__ void packObjectsToBuffer(int nObj, const MapEntry *map, const size_t *offsetsBytes,
                                    const int *offsets, const T *srcData, char *buffer)
{
    int objId = threadIdx.x + blockIdx.x * blockDim.x;

    if (objId >= nObj) return;
    
    auto m = map[objId];
    int buffId = m.getBufId();
    int  srcId = m.getId();
    T *dstData = (T*) (buffer + offsetsBytes[buffId]);
    int bufOffset = offsets[buffId];

    int dstId = objId - bufOffset;
    dstData[dstId] = srcData[srcId]; // TODO shift
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
        dstData[j] = srcData[j]; // TODO shift
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

    dstData[objId] = srcData[objId]; // TODO shift
}

} // namespace ObjectPackerKernels

ObjectPacker::ObjectPacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate) :
    Packer(pv, lpv, predicate),    
    ov(dynamic_cast<ObjectVector*>(pv)),
    lov(dynamic_cast<LocalObjectVector*>(lpv))
{
    if (ov == nullptr)
        die("object packers must work with object vectors");

    if (lov == nullptr)
        die("object packers must work with local object vectors");
}

size_t ObjectPacker::getPackedSizeBytes(int nobj)
{
    auto packedSizeParts = _getPackedSizeBytes(lov->dataPerParticle, nobj * ov->objSize);
    auto packedSizeObjs  = _getPackedSizeBytes(lov->dataPerObject,   nobj);

    return packedSizeParts + packedSizeObjs;
}

void ObjectPacker::packToBuffer(const DeviceBuffer<MapEntry>& map, const PinnedBuffer<int>& sizes,
                                const PinnedBuffer<int>& offsets, char *buffer, cudaStream_t stream)
{
    auto& partManager = lov->dataPerParticle;
    auto& objManager  = lov->dataPerObject;

    offsetsBytes.resize_anew(offsets.size());
    offsetsBytes.clear(stream);

    // pack particle data
    for (const auto& name_desc : partManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto packChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nObj = map.size();
            const int nthreads = 128;

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::packParticlesToBuffer,
                nObj, nthreads, 0, stream,
                map.devPtr(), ov->objSize, offsetsBytes.devPtr(), offsets.devPtr(),
                pinnedBuffPtr->devPtr(), buffer);

            updateOffsets<T>(sizes.size(), ov->objSize, sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(packChannel, desc->varDataPtr);
    }

    // pack object data
    for (const auto& name_desc : objManager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto packChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nObj = map.size();
            const int nthreads = 32;

            SAFE_KERNEL_LAUNCH(
                ObjectPackerKernels::packObjectsToBuffer,
                getNblocks(nObj, nthreads), nthreads, 0, stream,
                nObj, map.devPtr(), offsetsBytes.devPtr(), offsets.devPtr(),
                pinnedBuffPtr->devPtr(), buffer);

            updateOffsets<T>(sizes.size(), sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(packChannel, desc->varDataPtr);
    }
}

void ObjectPacker::unpackFromBuffer(const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                                    const char *buffer, cudaStream_t stream)
{
    
}
