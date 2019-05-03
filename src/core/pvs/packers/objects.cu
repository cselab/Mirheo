#include "objects.h"

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
                                PinnedBuffer<size_t>& offsetsBytes, char *buffer, cudaStream_t stream)
{
    
}

void ObjectPacker::unpackFromBuffer(PinnedBuffer<size_t>& offsetsBytes,
                                    const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                                    const char *buffer, cudaStream_t stream)
{
    
}
