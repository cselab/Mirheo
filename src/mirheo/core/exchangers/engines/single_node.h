#pragma once

#include "interface.h"

#include <mpi.h>
#include <string>

namespace mirheo
{

class ExchangeEntity;

/**
 * Engine used when there is only one node
 *
 * Simply swap senfBuf and recvBuf
 */
class SingleNodeEngine : public ExchangeEngine
{
public:
    SingleNodeEngine(std::unique_ptr<Exchanger>&& exchanger);
    ~SingleNodeEngine();
    
    void init    (cudaStream_t stream) override;
    void finalize(cudaStream_t stream) override;

private:
    void _copySend2Recv(ExchangeEntity *helper, cudaStream_t stream);
};

} // namespace mirheo
