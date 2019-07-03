#pragma once

#include "../exchanger_interfaces.h"

#include <mpi.h>
#include <string>

class ExchangeHelper;

/**
 * Engine implementing MPI exchange logic.
 *
 * The pipeline is as follows:
 * - base method init() sets up the communication:
 *   - calls base method postRecvSize() that issues MPI_Irecv() calls
 *     for sizes of data that has to be received
 *   - calls exchanger method prepareData() that fills the corresponding
 *     ExchangeHelper buffers with the data to exchange
 * - base method finalize() runs the communication (it could
 *   be split into send/recv pair, maybe this will be done later):
 *   - calls base method send() that sends the data from ExchangeHelper
 *     buffers to the relevant MPI processes
 *   - calls base method recv() that blocks until the sizes of the
 *     data and data themselves are received and stored in the ExchangeHelper
 *   - calls exchanger combineAndUploadData() that takes care
 *     of storing data from the ExchangeHelper to where is has to be
 */
class MPIExchangeEngine : public ExchangeEngine
{
public:
    MPIExchangeEngine(std::unique_ptr<Exchanger> exchanger, MPI_Comm comm, bool gpuAwareMPI);
    ~MPIExchangeEngine();
    
    void init(cudaStream_t stream)     override;
    void finalize(cudaStream_t stream) override;

private:
    std::unique_ptr<Exchanger> exchanger;
    
    std::vector<int> dir2rank;
    std::vector<int> dir2sendTag;
    std::vector<int> dir2recvTag;    
    int nActiveNeighbours;

    int myrank;
    MPI_Comm haloComm;
    bool gpuAwareMPI;
    int singleCopyThreshold = 4000000;

    void postRecvSize(ExchangeHelper *helper);
    void sendSizes(ExchangeHelper *helper);
    void postRecv(ExchangeHelper *helper);
    void wait(ExchangeHelper *helper, cudaStream_t stream);
    void send(ExchangeHelper *helper, cudaStream_t stream);
};
