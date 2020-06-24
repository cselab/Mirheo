// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

#include <mpi.h>
#include <string>

namespace mirheo
{

class ExchangeEntity;

/** \brief Special engine optimized for single node simulations.

    Instead of communicating thedata through MPI, the send and recv buffers are simply swapped.
 */
class SingleNodeExchangeEngine : public ExchangeEngine
{
public:
    /** \brief Construct a SingleNodeExchangeEngine.
        \param exchanger The class responsible to pack and unpack the data.
     */
    SingleNodeExchangeEngine(std::unique_ptr<Exchanger>&& exchanger);
    ~SingleNodeExchangeEngine();

    void init    (cudaStream_t stream) override;
    void finalize(cudaStream_t stream) override;

private:
    void _copySend2Recv(ExchangeEntity *helper, cudaStream_t stream);
};

} // namespace mirheo
