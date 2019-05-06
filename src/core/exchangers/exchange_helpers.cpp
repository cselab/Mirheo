#include "exchange_helpers.h"
#include "packers/interface.h"

template <typename T>
static void prefixSum(const PinnedBuffer<T>& sz, PinnedBuffer<T>& of)
{
    int n = sz.size();
    if (n == 0) return;

    of[0] = 0;
    for (int i = 0; i < n; i++)
        of[i+1] = of[i] + sz[i];
}

static void computeSizesBytes(const Packer *packer, const PinnedBuffer<int>& sz, PinnedBuffer<size_t>& szBytes)
{
    for (int i = 0; i < sz.size(); ++i)
        szBytes[i] = packer->getPackedSizeBytes(sz[i]);
}

ExchangeHelper::ExchangeHelper(std::string name, int uniqueId, Packer *packer) :
    name(name),
    uniqueId(uniqueId),
    packer(packer)
{
    recv.sizes       .resize_anew(nBuffers);
    recv.sizesBytes  .resize_anew(nBuffers);
    recv.offsets     .resize_anew(nBuffers+1);
    recv.offsetsBytes.resize_anew(nBuffers+1);
    
    send.sizes       .resize_anew(nBuffers);
    send.sizesBytes  .resize_anew(nBuffers);
    send.offsets     .resize_anew(nBuffers+1);
    send.offsetsBytes.resize_anew(nBuffers+1);
}

ExchangeHelper::~ExchangeHelper() = default;

void ExchangeHelper::computeRecvOffsets()
{
    prefixSum(recv.sizes, recv.offsets);
    computeSizesBytes(packer, recv.sizes, recv.sizesBytes);
    prefixSum(recv.sizesBytes, recv.offsetsBytes);
}

void ExchangeHelper::computeSendOffsets()
{
    prefixSum(send.sizes, send.offsets);
    computeSizesBytes(packer, send.sizes, send.sizesBytes);
    prefixSum(send.sizesBytes, send.offsetsBytes);
}

void ExchangeHelper::computeSendOffsets_Dev2Dev(cudaStream_t stream)
{
    send.sizes.downloadFromDevice(stream);
    computeSendOffsets();
    send.offsets.uploadToDevice(stream);
    send.offsetsBytes.uploadToDevice(stream);
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
    return uniqueId;
}

BufferOffsetsSizesWrap ExchangeHelper::wrapSendData()
{
    return {nBuffers, send.buffer.devPtr(), send.offsets.devPtr(), send.sizes.devPtr()};
}
