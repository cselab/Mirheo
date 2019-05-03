#include "particles.h"

#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <type_traits>

namespace ParticlePackerKernels
{

template <typename T>
__global__ void updateOffsets(int n, const int *sizes, size_t *offsetsBytes)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n) return;
    
    size_t sz = Packer::getPackedSize<T>(sizes[i]);
    offsetsBytes[i] += sz;
}

template <typename T>
__global__ void packToBuffer(int n, const MapEntry *map, const size_t *offsetsBytes, const T *srcData, char *buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n) return;

    auto m = map[i];
    int buffId = m.getBufId();
    int  srcId = m.getId();

    T *dstData = (T*) (buffer + offsetsBytes[buffId]);

    dstData[i] = srcData[srcId]; // TODO shift
}

template <typename T>
__global__ void unpackFromBuffer(int nBuffers, const int *offsets, int n, const char *buffer, const size_t *offsetsBytes, T *dstData)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n) return;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();
    
    int buffId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, i);
    int pid = i - sharedOffsets[buffId];
    
    const T *srcData = (const T*) (buffer + offsetsBytes[buffId]);

    dstData[pid] = srcData[pid]; // TODO shift
}



} // namespace ParticlePackerKernels

ParticlePacker::ParticlePacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate) :
    Packer(pv, lpv, predicate)
{}

size_t ParticlePacker::getPackedSizeBytes(int n)
{
    return _getPackedSizeBytes(lpv->dataPerParticle, n);
}

void ParticlePacker::packToBuffer(const DeviceBuffer<MapEntry>& map, const PinnedBuffer<int>& sizes,
                                  PinnedBuffer<size_t>& offsetsBytes, char *buffer, cudaStream_t stream)
{
    auto& manager = lpv->dataPerParticle;

    for (const auto& name_desc : manager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto packChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            {
                int n = map.size();
                const int nthreads = 128;

                SAFE_KERNEL_LAUNCH(
                    ParticlePackerKernels::packToBuffer,
                    getNblocks(n, nthreads), nthreads, 0, stream,
                    n, map.devPtr(), offsetsBytes.devPtr(),
                    pinnedBuffPtr->devPtr(), buffer);
            }
            {
                int n = sizes.size();
                const int nthreads = 32;

                SAFE_KERNEL_LAUNCH(
                    ParticlePackerKernels::updateOffsets<T>,
                    getNblocks(n, nthreads), nthreads, 0, stream,
                    n, sizes.devPtr(), offsetsBytes.devPtr());
            }
        };
        
        mpark::visit(packChannel, desc->varDataPtr);
    }
}

void ParticlePacker::unpackFromBuffer(PinnedBuffer<size_t>& offsetsBytes,
                                      const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                                      const char *buffer, cudaStream_t stream)
{
    auto& manager = lpv->dataPerParticle;

    for (const auto& name_desc : manager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto unpackChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            int nBuffers = sizes.size();
            {
                int n = offsets[nBuffers];
                const int nthreads = 128;

                SAFE_KERNEL_LAUNCH(
                    ParticlePackerKernels::unpackFromBuffer,
                    getNblocks(n, nthreads), nthreads, 0, stream,
                    nBuffers, offsets.devPtr(), n, buffer,
                    offsetsBytes.devPtr(), pinnedBuffPtr->devPtr());
            }
            {
                const int nthreads = 32;

                SAFE_KERNEL_LAUNCH(
                    ParticlePackerKernels::updateOffsets<T>,
                    getNblocks(nBuffers, nthreads), nthreads, 0, stream,
                    nBuffers, sizes.devPtr(), offsetsBytes.devPtr());
            }
        };
        
        mpark::visit(unpackChannel, desc->varDataPtr);
    }
}
