// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "single_node.h"
#include "../exchange_entity.h"

#include <mirheo/core/exchangers/interface.h>
#include <mirheo/core/logger.h>

#include <algorithm>

namespace mirheo
{

SingleNodeExchangeEngine::SingleNodeExchangeEngine(std::unique_ptr<Exchanger>&& exchanger) :
    ExchangeEngine(std::move(exchanger))
{}

SingleNodeExchangeEngine::~SingleNodeExchangeEngine() = default;

void SingleNodeExchangeEngine::init(cudaStream_t stream)
{
    const size_t numExchangeEntities = exchanger_->getNumExchangeEntities();

    for (size_t i = 0; i < numExchangeEntities; ++i)
        if (!exchanger_->needExchange(i))
            debug("Exchange of PV '%s' is skipped", exchanger_->getExchangeEntity(i)->getCName());

    // Derived class determines what to send
    for (size_t i = 0; i < numExchangeEntities; ++i)
        if (exchanger_->needExchange(i))
            exchanger_->prepareSizes(i, stream);

    CUDA_Check( cudaStreamSynchronize(stream) );

    // Derived class determines what to send
    for (size_t i = 0; i < numExchangeEntities; ++i)
        if (exchanger_->needExchange(i))
            exchanger_->prepareData(i, stream);
}

void SingleNodeExchangeEngine::finalize(cudaStream_t stream)
{
    const size_t numExchangeEntities = exchanger_->getNumExchangeEntities();

    for (size_t i = 0; i < numExchangeEntities; ++i)
        if (exchanger_->needExchange(i))
            _copySend2Recv(exchanger_->getExchangeEntity(i), stream);

    for (size_t i = 0; i < numExchangeEntities; ++i)
        if (exchanger_->needExchange(i))
            exchanger_->combineAndUploadData(i, stream);
}


void SingleNodeExchangeEngine::_copySend2Recv(ExchangeEntity *helper, cudaStream_t stream)
{
    const auto bulkId = helper->bulkId;

    if (helper->send.sizes[bulkId] != 0)
        error("Non-empty message to itself detected, this may fail with the Single node engine, "
              "working with particle vector '%s'", helper->getCName());

    // copy (not swap) as we may need sizes from other classes
    helper->recv.sizes       .copy(helper->send.sizes,        stream);
    helper->recv.offsets     .copy(helper->send.offsets,      stream);
    helper->recv.sizesBytes  .copy(helper->send.sizesBytes,   stream);
    helper->recv.offsetsBytes.copy(helper->send.offsetsBytes, stream);

    std::swap(helper->recv.buffer,      helper->send.buffer);
}


} // namespace mirheo
