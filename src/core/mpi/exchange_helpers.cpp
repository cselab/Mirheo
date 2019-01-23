#include "exchange_helpers.h"

ExchangeHelper::ExchangeHelper(std::string name, int datumSize) :
    name(name),
    datumSize(datumSize)
{
    recvSizes.  resize_anew(nBuffers);
    recvOffsets.resize_anew(nBuffers+1);
    
    sendSizes.  resize_anew(nBuffers);
    sendOffsets.resize_anew(nBuffers+1);
}

ExchangeHelper::~ExchangeHelper() = default;

void ExchangeHelper::setDatumSize(int size)
{
    datumSize = size;
}

void ExchangeHelper::computeRecvOffsets()
{
    computeOffsets(recvSizes, recvOffsets);
}

void ExchangeHelper::computeSendOffsets()
{
    computeOffsets(sendSizes, sendOffsets);
}

void ExchangeHelper::computeSendOffsets_Dev2Dev(cudaStream_t stream)
{
    sendSizes.downloadFromDevice(stream);
    computeSendOffsets();
    sendOffsets.uploadToDevice(stream);
}

void ExchangeHelper::resizeSendBuf()
{
    sendBuf.resize_anew(sendOffsets[nBuffers] * datumSize);
}

void ExchangeHelper::resizeRecvBuf()
{
    recvBuf.resize_anew(recvOffsets[nBuffers] * datumSize);
}


BufferOffsetsSizesWrap ExchangeHelper::wrapSendData()
{
    return {nBuffers, sendBuf.devPtr(), sendOffsets.devPtr(), sendSizes.devPtr()};
}

void ExchangeHelper::computeOffsets(const PinnedBuffer<int>& sz, PinnedBuffer<int>& of)
{
    int n = sz.size();
    if (n == 0) return;
    
    of[0] = 0;
    for (int i = 0; i < n; i++)
        of[i+1] = of[i] + sz[i];
}
