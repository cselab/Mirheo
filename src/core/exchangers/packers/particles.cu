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
    if (i >= n) return;

    auto m = map[i];
    int bufId = m.getBufId();
    int srcId = m.getId();

    auto dstData = reinterpret_cast<T*>(buffer + offsetsBytes[bufId]);
    int dstId = i - offsets[bufId];

    dstData[dstId] = shift(srcData[srcId], bufId);
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

    if (i >= n) return;
    
    int bufId = dispatchThreadsPerBuffer(nBuffers, sharedOffsets, i);
    int pid = i - sharedOffsets[bufId];
    
    auto srcData = reinterpret_cast<const T*> (buffer + offsetsBytes[bufId]);

    dstData[i] = srcData[pid];
}

} // namespace ParticlePackerKernels

ParticlesPacker::ParticlesPacker(ParticleVector *pv, PackPredicate predicate) :
    Packer(pv, predicate)
{}

size_t ParticlesPacker::getPackedSizeBytes(int n) const
{
    return _getPackedSizeBytes(pv->local()->dataPerParticle, n);
}

void ParticlesPacker::packToBuffer(const LocalParticleVector *lpv, const DeviceBuffer<MapEntry>& map,
                                   BufferInfos *helper, const std::vector<std::string>& alreadyPacked, cudaStream_t stream)
{
    auto& manager = lpv->dataPerParticle;

    int nBuffers = helper->sizes.size();
    
    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    // advance offsets to skip the already packed data
    for (auto name : alreadyPacked)
    {
        auto& desc = manager.getChannelDescOrDie(name);
        auto advanceOffset = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;
            updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        mpark::visit(advanceOffset, desc.varDataPtr);
    }
    
    for (const auto& name_desc : manager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        bool isAlreadPacked = std::find(alreadyPacked.begin(), alreadyPacked.end(), name_desc.first) != alreadyPacked.end();
        if (isAlreadPacked) continue;

        Shifter shift(desc->shiftTypeSize > 0, pv->state->domain);

        auto packChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            int n = map.size();
            const int nthreads = 128;

            SAFE_KERNEL_LAUNCH(
                ParticlePackerKernels::packToBuffer,
                getNblocks(n, nthreads), nthreads, 0, stream,
                n, map.devPtr(), offsetsBytes.devPtr(), helper->offsets.devPtr(),
                pinnedBuffPtr->devPtr(), shift, helper->buffer.devPtr());

            updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(packChannel, desc->varDataPtr);
    }
}

void ParticlesPacker::unpackFromBuffer(LocalParticleVector *lpv, const BufferInfos *helper, int oldSize, cudaStream_t stream)
{
    auto& manager = lpv->dataPerParticle;

    offsetsBytes.copyFromDevice(helper->offsetsBytes, stream);

    int nBuffers  = helper->sizes.size();
    int nIncoming = helper->offsets[nBuffers];
    
    for (const auto& name_desc : manager.getSortedChannels())
    {
        if (!predicate(name_desc)) continue;
        auto& desc = name_desc.second;

        debug2("unpacking particle channel '%s' of pv '%s'", name_desc.first.c_str(), pv->name.c_str());

        auto unpackChannel = [&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            const int nthreads = 128;
            const size_t sharedMem = nBuffers * sizeof(int);

            SAFE_KERNEL_LAUNCH(
                ParticlePackerKernels::unpackFromBuffer,
                getNblocks(nIncoming, nthreads), nthreads, sharedMem, stream,
                nBuffers, helper->offsets.devPtr(), nIncoming, helper->buffer.devPtr(),
                offsetsBytes.devPtr(), pinnedBuffPtr->devPtr() + oldSize);

            updateOffsets<T>(nBuffers, helper->sizes.devPtr(), offsetsBytes.devPtr(), stream);
        };
        
        mpark::visit(unpackChannel, desc->varDataPtr);
    }
}
