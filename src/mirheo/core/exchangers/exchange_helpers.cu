#include "exchange_helpers.h"

#include <mirheo/core/pvs/packers/rods.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace ExchangeHelpersKernels
{
// must be executed with only one warp
template <class Packer>
__global__ void computeOffsetsSizeBytes(BufferOffsetsSizesWrap wrapData, size_t *sizesBytes,
                                        Packer packer)
{
    const int tid = threadIdx.x;
    assert(tid < warpSize);

    int size = 0;

    if (tid < wrapData.nBuffers)
        size = wrapData.sizes[tid];

    size_t sizeBytes = packer.getSizeBytes(size);
    
    int    offset      = warpExclusiveScan(size     );
    size_t offsetBytes = warpExclusiveScan(sizeBytes);

    if (tid < wrapData.nBuffers + 1)
    {
        wrapData.offsets     [tid] = offset;
        wrapData.offsetsBytes[tid] = offsetBytes;
        sizesBytes[tid] = sizeBytes;
    }
}

} // namespace ExchangeHelpersKernels


template <class Packer>
inline void computeOffsetsSizeBytesDev(const BufferOffsetsSizesWrap& wrapData,
                                       PinnedBuffer<size_t>& sizeBytes,
                                       const Packer& packer, cudaStream_t stream)
{
    // must be launched on one warp only
    constexpr int nthreads = 32;
    constexpr int nblocks  = 1;
    
    SAFE_KERNEL_LAUNCH(
        ExchangeHelpersKernels::computeOffsetsSizeBytes,
        nblocks, nthreads, 0, stream,
        wrapData, sizeBytes.devPtr(), packer);
}

static void computeOffsetsSizeBytesDev(const BufferOffsetsSizesWrap& wrapData,
                                       PinnedBuffer<size_t>& sizeBytes,
                                       ParticlePacker *pp, cudaStream_t stream)
{
    auto rp = dynamic_cast<RodPacker*>(pp);
    auto op = dynamic_cast<ObjectPacker*>(pp);

    auto execute = [&](const auto& packerHandler)
    {
        computeOffsetsSizeBytesDev(wrapData, sizeBytes, packerHandler, stream);
    };
    
    if      (rp != nullptr) execute(rp->handler());
    else if (op != nullptr) execute(op->handler());
    else                    execute(pp->handler());
}

template <typename T>
static void prefixSum(const PinnedBuffer<T>& sz, PinnedBuffer<T>& of)
{
    const size_t n = sz.size();
    if (n == 0) return;

    of[0] = 0;
    for (size_t i = 0; i < n; i++)
        of[i+1] = of[i] + sz[i];
}

static void computeSizesBytes(const ParticlePacker *packer, const PinnedBuffer<int>& sz, PinnedBuffer<size_t>& szBytes)
{
    for (size_t i = 0; i < sz.size(); ++i)
        szBytes[i] = packer->getSizeBytes(sz[i]);
}

void BufferInfos::clearAllSizes(cudaStream_t stream)
{
    sizes.clear(stream);
    sizesBytes.clear(stream);
}

void BufferInfos::resizeInfos(int nBuffers)
{
    sizes        .resize_anew(nBuffers);
    sizesBytes   .resize_anew(nBuffers);
    offsets      .resize_anew(nBuffers+1);
    offsetsBytes .resize_anew(nBuffers+1);
}

void BufferInfos::uploadInfosToDevice(cudaStream_t stream)
{
    sizes        .uploadToDevice(stream);
    sizesBytes   .uploadToDevice(stream);
    offsets      .uploadToDevice(stream);
    offsetsBytes .uploadToDevice(stream);
}

char* BufferInfos::getBufferDevPtr(int bufId)
{
    return buffer.devPtr() + offsetsBytes[bufId];
}

ExchangeHelper::ExchangeHelper(std::string name, int uniqueId, ParticlePacker *packer) :
    name_(name),
    uniqueId_(uniqueId),
    packer_(packer)
{
    recv.resizeInfos(nBuffers);
    send.resizeInfos(nBuffers);
}

ExchangeHelper::~ExchangeHelper() = default;

void ExchangeHelper::computeRecvOffsets()
{
    prefixSum(recv.sizes, recv.offsets);
    computeSizesBytes(packer_, recv.sizes, recv.sizesBytes);
    prefixSum(recv.sizesBytes, recv.offsetsBytes);
}

void ExchangeHelper::computeSendOffsets()
{
    prefixSum(send.sizes, send.offsets);
    computeSizesBytes(packer_, send.sizes, send.sizesBytes);
    prefixSum(send.sizesBytes, send.offsetsBytes);
}

void ExchangeHelper::computeSendOffsets_Dev2Dev(cudaStream_t stream)
{
    computeOffsetsSizeBytesDev(wrapSendData(), send.sizesBytes, packer_, stream);
    
    send.sizes       .downloadFromDevice(stream, ContainersSynch::Asynch);
    send.offsets     .downloadFromDevice(stream, ContainersSynch::Asynch);
    send.sizesBytes  .downloadFromDevice(stream, ContainersSynch::Asynch);
    send.offsetsBytes.downloadFromDevice(stream, ContainersSynch::Synch);
}

void ExchangeHelper::resizeSendBuf()
{
    auto size = send.offsetsBytes[nBuffers];
    send.buffer.resize_anew(size);
}

void ExchangeHelper::resizeRecvBuf()
{
    auto size = recv.offsetsBytes[nBuffers];
    recv.buffer.resize_anew(size);
}

int ExchangeHelper::getUniqueId() const
{
    return uniqueId_;
}

BufferOffsetsSizesWrap ExchangeHelper::wrapSendData()
{
    return {nBuffers, send.buffer.devPtr(),
            send.offsets.devPtr(), send.sizes.devPtr(),
            send.offsetsBytes.devPtr()};
}

BufferOffsetsSizesWrap ExchangeHelper::wrapRecvData()
{
    return {nBuffers, recv.buffer.devPtr(),
            recv.offsets.devPtr(), recv.sizes.devPtr(),
            recv.offsetsBytes.devPtr()};
}

const std::string& ExchangeHelper::getName() const {return name_;}
const char* ExchangeHelper::getCName() const {return name_.c_str();}

} // namespace mirheo
