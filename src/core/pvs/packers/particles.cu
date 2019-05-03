#include "particles.h"

#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <type_traits>

namespace ParticlePackerKernels
{

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

} // namespace ParticlePackerKernels

ParticlePacker::ParticlePacker(ParticleVector *pv, LocalParticleVector *lpv, PackPredicate predicate) :
    Packer(pv, lpv, predicate)
{}

size_t ParticlePacker::getPackedSizeBytes(int n)
{
    return _getPackedSizeBytes(lpv->dataPerParticle, n);
}

void ParticlePacker::packToBuffer(DeviceBuffer<MapEntry>& map, PinnedBuffer<size_t>& offsets, PinnedBuffer<int>& sizes, char *buffer, cudaStream_t stream)
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
                    n, map.devPtr(), offsets.devPtr(),
                    pinnedBuffPtr->devPtr(), buffer);
            }
            {
                int n = sizes.size();
                const int nthreads = 32;

                SAFE_KERNEL_LAUNCH(
                    ParticlePackerKernels::updateOffsets<T>,
                    getNblocks(n, nthreads), nthreads, 0, stream,
                    n, sizes.devPtr(), offsets.devPtr());
            }
        };
        
        mpark::visit(packChannel, desc->varDataPtr);
    }
}
