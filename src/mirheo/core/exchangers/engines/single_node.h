#pragma once

#include "../exchanger_interfaces.h"

#include <mpi.h>
#include <string>

namespace mirheo
{

class ExchangeHelper;

/**
 * Engine used when there is only one node
 *
 * Simply swap senfBuf and recvBuf
 */
class SingleNodeEngine : public ExchangeEngine
{
public:
    SingleNodeEngine(std::unique_ptr<Exchanger> exchanger);
    ~SingleNodeEngine();
    
    void init(cudaStream_t stream)     override;
    void finalize(cudaStream_t stream) override;

private:
    std::unique_ptr<Exchanger> exchanger_;
    
    void copySend2Recv(ExchangeHelper *helper, cudaStream_t stream);
};

} // namespace mirheo
