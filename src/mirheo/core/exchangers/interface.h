#pragma once

#include <mirheo/core/datatypes.h>

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace mirheo
{

class ExchangeEntity;

/**
 * Interface for classes preparing and packing particles for exchange
 * 
 * The virtual method prepareSizes() determines the size
 * of data to be exchanged
 *
 * The virtual method prepareData() fills the corresponding
 * ExchangeEntity buffers with the data to exchange
 *
 * The virtual combineAndUploadData() takes care
 * of storing data from the ExchangeEntity to where is has to be
 */
class Exchanger
{
public:

    virtual ~Exchanger();

    void addExchangeEntity(std::unique_ptr<ExchangeEntity>&& e);

    ExchangeEntity*       getExchangeEntity(size_t id);
    const ExchangeEntity* getExchangeEntity(size_t id) const;

    size_t getNumExchangeEntities() const;
    
    /**
     * This function has to provide sizes of the data to be communicated in
     * `helpers[id].`. It has to set `helpers[id].sendSizes` and
     * `helpers[id].sendOffsets` on the CPU
     *
     * @param id helper id that will be filled with data
     */
    virtual void prepareSizes(size_t id, cudaStream_t stream) = 0;

    /**
     * This function has to provide data that has to be communicated in
     * `helpers[id]`. It has to set `helpers[id].sendSizes` and
     * `helpers[id].sendOffsets` on the CPU, but the bulk data of
     * `helpers[id].sendBuf` must be only set on GPU. The reason is because
     * in most cases offsets and sizes are anyways needed on CPU side
     * to resize stuff, but bulk data are not; and it would be possible
     * to change the MPI backend to CUDA-aware calls.
     *
     * @param id helper id that will be filled with data
     */
    virtual void prepareData (size_t id, cudaStream_t stream) = 0;

    /**
     * This function has to unpack the received data. Similarly to
     * prepareData() function, when it is called `helpers[id].recvSizes`
     * and `helpers[id].recvOffsets` are set according to the
     * received data on the CPU only. However, `helpers[id].recvBuf`
     * will contain already GPU data
     *
     * @param id helper id that is filled with the received data
     */
    virtual void combineAndUploadData(size_t id, cudaStream_t stream) = 0;

    /**
     * If the ParticleVector didn't change since the last similar MPI
     * exchange, there is no need to run the exchange again. This function
     * controls such behaviour
     * @param id of the ParticleVector and associated ExchangeEntity
     * @return true if exchange is required, false - if not
     */
    virtual bool needExchange(size_t id) = 0;

private:
    /**
     * Vector of helpers, that have buffers for data exchange
     * and other required information, see :any:`ExchangeEntity`
     */ 
    std::vector<std::unique_ptr<ExchangeEntity>> helpers_;
};

} // namespace mirheo
