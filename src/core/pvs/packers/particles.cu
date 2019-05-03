#include "particles.h"

#include <core/pvs/particle_vector.h>

NAMESPACE_BEGIN(ParticlePackerKernels)

template <typename T>
__global__ void updateOffsets(int n, const int *sizes, size_t *offsets)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n) return;
    
    size_t sz = Packer::getPackedSize<T>(sizes[i]);
    offsets[i] += sz;
}

template <typename T>
__global__ void packToBuffer(int n, const MapEntry *map, const size_t *offsets, const T *srcData, char *buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n) return;

    auto m = map[i];
    int buffId = m.getBufId();
    int  srcId = m.getId();

    T *dstAddrBase = (T*) (buffer + offsets[buffId]);

    dstAddrBase[i] = srcData[srcId]; // TODO shift
}

NAMESPACE_END(ParticlePackerKernels)

ParticlePacker::ParticlePacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate) :
    Packer(pv, lpv, predicate)
{}

size_t ParticlePacker::getPackedSizeBytes(int n)
{
    return _getPackedSizeBytes(lpv->dataPerParticle, n);
}

void ParticlePacker::packToBuffer(const MapEntry *map, PinnedBuffer<size_t> offsets, PinnedBuffer<int> sizes, char *buufer, cudaStream_t stream)
{
    // for (const auto& name_desc : manager.getSortedChannels())
    // {
    //     // TODO
    // }
}
