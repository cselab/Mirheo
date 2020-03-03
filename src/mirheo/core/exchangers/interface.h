#pragma once

#include <mirheo/core/datatypes.h>

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace mirheo
{

class ExchangeEntity;

/** \brief Pack and unpack \c ParticleVector objects for exchange.

    The user should register one (or more) ExchangeEntity objects that represent
    the data to exchange.
    The functions interface functions can then be called in the correct order to pack and unpack the data.
    
    Designed to be used with an ExchangeEngine.
 */
class Exchanger
{
public:
    virtual ~Exchanger();

    /** \brief register an ExchangeEntity in this exchanger.
        \param [in] e The ExchangeEntity object to register. Will pass ownership.
     */
    void addExchangeEntity(std::unique_ptr<ExchangeEntity>&& e);

    /// \return ExchangeEntity with the given id (0 <= id < getNumExchangeEntities()).
    ExchangeEntity*       getExchangeEntity(size_t id);
    const ExchangeEntity* getExchangeEntity(size_t id) const; ///< see getExchangeEntity()

    /// \return The number of registered ExchangeEntity.
    size_t getNumExchangeEntities() const;
    
    /** \brief Compute the sizes of the data to be communicated in the given ExchangeEntity.
        \param id The index of the concerned ExchangeEntity
        \param stream Execution stream

        After this call, the `send.sizes`, `send.sizeBytes`, `send.offsets` and `send.offsetsBytes` 
        of the ExchangeEntity are available on the CPU.
     */
    virtual void prepareSizes(size_t id, cudaStream_t stream) = 0;

    /** \brief Pack the data managed by the given ExchangeEntity
        \param id The index of the concerned ExchangeEntity
        \param stream Execution stream
        \note Must be executed after prepareSizes()
     */
    virtual void prepareData (size_t id, cudaStream_t stream) = 0;

    /** \brief Unpack the received data. 
        \param id The index of the concerned ExchangeEntity
        \param stream Execution stream

        After this call, the `recv.sizes`, `recv.sizeBytes`, `recv.offsets` and `recv.offsetsBytes` 
        of the ExchangeEntity must be available on the CPU and GPU before this call.
        Furthermore, the recv buffers must already be on the device memory.
        
        \note Must be executed after prepareData()
     */
    virtual void combineAndUploadData(size_t id, cudaStream_t stream) = 0;

    /** \brief Stats if the data of an ExchangeEntity needs to be exchanged.
        \param id The index of the concerned ExchangeEntity
        \return \c true if exchange is required, \c false otherwise

        If the ParticleVector didn't change since the last exchange, 
        there is no need to run the exchange again. 
        This function controls such behaviour.
     */
    virtual bool needExchange(size_t id) = 0;

private:
    /// list of ExchangeEntity that manages the data to exchange.
    std::vector<std::unique_ptr<ExchangeEntity>> helpers_;
};

} // namespace mirheo
