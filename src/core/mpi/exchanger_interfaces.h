#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>

class ExchangeHelper;

/**
 * Interface for classes preparing and packing particles for exchange
 * 
 * The virtual method prepareSizes() determines the size
 * of data to be exchanged
 *
 * The virtual method prepareData() fills the corresponding
 * ExchangeHelper buffers with the data to exchange
 *
 * The virtual combineAndUploadData() takes care
 * of storing data from the ExchangeHelper to where is has to be
 */
class ParticleExchanger
{
public:

    virtual ~ParticleExchanger();

    /**
     * Vector of helpers, that have buffers for data exchange
     * and other required information, see :any:`ExchangeHelper`
     */ 
    std::vector<std::unique_ptr<ExchangeHelper>> helpers;

    /**
     * This function has to provide sizes of the data to be communicated in
     * `helpers[id].`. It has to set `helpers[id].sendSizes` and
     * `helpers[id].sendOffsets` on the CPU
     *
     * @param id helper id that will be filled with data
     */
    virtual void prepareSizes(int id, cudaStream_t stream) = 0;

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
    virtual void prepareData (int id, cudaStream_t stream) = 0;

    /**
     * This function has to unpack the received data. Similarly to
     * prepareData() function, when it is called `helpers[id].recvSizes`
     * and `helpers[id].recvOffsets` are set according to the
     * received data on the CPU only. However, `helpers[id].recvBuf`
     * will contain already GPU data
     *
     * @param id helper id that is filled with the received data
     */
    virtual void combineAndUploadData(int id, cudaStream_t stream) = 0;

    /**
     * If the ParticleVector didn't change since the last similar MPI
     * exchange, there is no need to run the exchange again. This function
     * controls such behaviour
     * @param id of the ParticleVector and associated ExchangeHelper
     * @return true if exchange is required, false - if not
     */
    virtual bool needExchange(int id) = 0;    
};



class ExchangeEngine
{
public:
    virtual void init(cudaStream_t stream)     = 0;
    virtual void finalize(cudaStream_t stream) = 0;
    virtual ~ExchangeEngine();
};
