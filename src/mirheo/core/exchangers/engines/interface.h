#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace mirheo
{
class Exchanger;

class ExchangeEngine
{
public:
    ExchangeEngine(std::unique_ptr<Exchanger>&& exchanger);
    virtual ~ExchangeEngine();
    
    virtual void init(cudaStream_t stream)     = 0;
    virtual void finalize(cudaStream_t stream) = 0;

protected:
    std::unique_ptr<Exchanger> exchanger_;
};

} // namespace mirheo
