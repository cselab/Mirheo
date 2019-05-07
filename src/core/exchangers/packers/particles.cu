#include "particles.h"
#include "common.h"
#include "shifter.h"

#include "../exchange_helpers.h"

#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <type_traits>

namespace ParticlePackerKernels
{
template <typename T>
__global__ void packToBuffer(int n, const MapEntry *map, const size_t *offsetsBytes, const int *offsets,
                             const T *srcData, Shifter shift, char *buffer)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > n) return;

    auto m = map[i];
    int buffId = m.getBufId();
    int  srcId = m.getId();

    T *dstData = (T*) (buffer + offsetsBytes[buffId]);
    int dstId = i - offsets[buffId];

    dstData[dstId] = shift(srcData[srcId], buffId);
}

template <typename T>
__global__ void unpackFromBuffer(int nBuffers, const int *offsets, int n, const char *buffer,
                                 const size_t *offsetsBytes, T *dstData)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ int sharedOffsets[];

    for (int i = threadIdx.x; i < nBuffers; i += blockDim.x)
        sharedOffsets[i] = offsets[i];
    __syncthreads();

    if (i > n) return;
    
    int buffId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, i);
    int pid = i - sharedOffsets[buffId];
    
    const T *srcData = (const T*) (buffer + offsetsBytes[buffId]);

    dstData[pid] = srcData[pid];
}

} // namespace ParticlePackerKernels

ParticlesPacker::ParticlesPacker(ParticleVector *pv, PackPredicate predicate) :
    Packer(pv, predicate)
{}

size_t ParticlesPacker::getPackedSizeBytes(int n) const
{
    return _getPackedSizeBytes(pv->local()->dataPerParticle, n);
}

void ParticlesPacker::packToBuffer(const LocalParticleVector *lpv, ExchangeHelper *helper, cudaStream_t stream)
{
    auto& manager = lpv->dataPerParticle;

    int nBuffers = helper->send.sizes.size();
    
    offsetsBytes.copyFromDevice(helper->send.offsetsBytes, stream);
    updateOffsets<float4>(nBuffers, helper->send.sizes.devPtr(), offsetsBytes.devPtr(), stream); // positions
    
    for (const auto& name_desc : manager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        Shifter shift(desc->shiftTypeSize > 0, pv->state->domain);

        auto packChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            int n = helper->map.size();
            const int nthreads = 128;

            SAFE_KERNEL_LAUNCH(
                ParticlePackerKernels::packToBuffer,
                getNblocks(n, nthreads), nthreads, 0, stream,
                n, helper->map.devPtr(), offsetsBytes.devPtr(), helper->send.offsets.devPtr(),
                pinnedBuffPtr->devPtr(), shift, helper->send.buffer.devPtr());

            updateOffsets<T>(nBuffers, helper->send.sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(packChannel, desc->varDataPtr);
    }
}

void ParticlesPacker::unpackFromBuffer(LocalParticleVector *lpv,
                                       const PinnedBuffer<int>& offsets, const PinnedBuffer<int>& sizes,
                                       const char *buffer, int oldSize, cudaStream_t stream)
{
    auto& manager = lpv->dataPerParticle;

    offsetsBytes.resize_anew(offsets.size());
    offsetsBytes.clear(stream);
    updateOffsets<float4>(sizes.size(), sizes.devPtr(), offsetsBytes.devPtr(), stream); // positions

    int nBuffers  = sizes.size();
    int nIncoming = offsets[nBuffers];
    
    for (const auto& name_desc : manager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        auto unpackChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nthreads = 128;
            const size_t sharedMem = nBuffers * sizeof(int);

            SAFE_KERNEL_LAUNCH(
                ParticlePackerKernels::unpackFromBuffer,
                getNblocks(nIncoming, nthreads), nthreads, sharedMem, stream,
                nBuffers, offsets.devPtr(), nIncoming, buffer,
                offsetsBytes.devPtr(), pinnedBuffPtr->devPtr() + oldSize);

            updateOffsets<T>(sizes.size(), sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(unpackChannel, desc->varDataPtr);
    }
}
