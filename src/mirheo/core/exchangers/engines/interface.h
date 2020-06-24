// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace mirheo
{
class Exchanger;

/** \brief Base communication engine class.

    Responsible to communicate the data managed by an \c Exchanger between different subdomains.
    The communication is split into two parts so that asynchronous communication can be used.
    Every init() call must have a single finalize() call that follows.
 */
class ExchangeEngine
{
public:
    /** Construct a communication engine.
        \param exchanger The \c Exchanger object that will prepare the data to communicate.
        The ownership of exchanger is transfered to the engine.
    */
    ExchangeEngine(std::unique_ptr<Exchanger>&& exchanger);
    virtual ~ExchangeEngine();

    /** \brief Initialize the communication.
        \param stream Execution stream used to prepare / download the data

        The data packing from the exchanger happens in this step.
     */
    virtual void init(cudaStream_t stream)     = 0;

    /** \brief Finalize the communication.
        \param stream Execution stream used to upload / unpack the data

        Must follow a pending init() call.
        The data unpacking from the exchanger happens in this step.
     */
    virtual void finalize(cudaStream_t stream) = 0;

protected:
    std::unique_ptr<Exchanger> exchanger_; ///< object that packs and unpacks the data
};

} // namespace mirheo
