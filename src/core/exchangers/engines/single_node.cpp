#include "single_node.h"
#include "../exchange_helpers.h"

#include <core/logger.h>
#include <algorithm>

SingleNodeEngine::SingleNodeEngine(std::unique_ptr<Exchanger> exchanger) :
        exchanger(std::move(exchanger))
{}

SingleNodeEngine::~SingleNodeEngine() = default;

void SingleNodeEngine::init(cudaStream_t stream)
{
    auto& helpers = exchanger->helpers;
    
    for (size_t i = 0; i < helpers.size(); ++i)
        if (!exchanger->needExchange(i))
            debug("Exchange of PV '%s' is skipped", helpers[i]->name.c_str());
    
    // Derived class determines what to send
    for (size_t i = 0; i < helpers.size(); ++i)
        if (exchanger->needExchange(i))
            exchanger->prepareSizes(i, stream);
        
    CUDA_Check( cudaStreamSynchronize(stream) );

    // Derived class determines what to send
    for (size_t i = 0; i < helpers.size(); ++i)
        if (exchanger->needExchange(i))
            exchanger->prepareData(i, stream);
}

void SingleNodeEngine::finalize(cudaStream_t stream)
{
    auto& helpers = exchanger->helpers;

    for (size_t i = 0; i < helpers.size(); ++i)
        if (exchanger->needExchange(i))
            copySend2Recv(helpers[i].get(), stream);
        
    for (size_t i = 0; i < helpers.size(); ++i)
        if (exchanger->needExchange(i))
            exchanger->combineAndUploadData(i, stream);
}


void SingleNodeEngine::copySend2Recv(ExchangeHelper *helper, cudaStream_t stream)
{
    auto bulkId = helper->bulkId;
    
    if (helper->send.sizes[bulkId] != 0)
        error("Non-empty message to itself detected, this may fail with the Single node engine, "
            "working with particle vector '%s'", helper->name.c_str());

    // copy (not swap) as we may need sizes from other classes
    helper->recv.sizes       .copy(helper->send.sizes,        stream);
    helper->recv.offsets     .copy(helper->send.offsets,      stream);
    helper->recv.sizesBytes  .copy(helper->send.sizesBytes,   stream);
    helper->recv.offsetsBytes.copy(helper->send.offsetsBytes, stream);

    std::swap(helper->recv.buffer,      helper->send.buffer);
}

