// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

#include <mpi.h>
#include <string>
#include <vector>

namespace mirheo
{

class ExchangeEntity;

/** \brief Engine implementing asynchronous MPI communication.

    The pipeline is as follows:
    - init() prepares the data into buffers, exchange the sizes, allocate recv buffers and post
      the asynchronous communication calls.
    - finalize() waits for the communication to finish and unpacks the data.
 */
class MPIExchangeEngine : public ExchangeEngine
{
public:
    /** \brief Construct a MPIExchangeEngine.
        \param exchanger The class responsible to pack and unpack the data.
        \param comm The cartesian communicator that represents the simulation domain.
        \param gpuAwareMPI \c true to enable RDMA implementation. Only works if the MPI library has this feature implemented.
     */
    MPIExchangeEngine(std::unique_ptr<Exchanger>&& exchanger, MPI_Comm comm, bool gpuAwareMPI);
    ~MPIExchangeEngine();

    void init(cudaStream_t stream)     override;
    void finalize(cudaStream_t stream) override;

private:
    void _postRecvSize(ExchangeEntity *helper);
    void _sendSizes   (ExchangeEntity *helper);
    void _postRecv    (ExchangeEntity *helper);
    void _wait        (ExchangeEntity *helper, cudaStream_t stream);
    void _send        (ExchangeEntity *helper, cudaStream_t stream);

private:
    std::vector<int> dir2rank_;
    std::vector<int> dir2sendTag_;
    std::vector<int> dir2recvTag_;

    bool gpuAwareMPI_;

    MPI_Comm haloComm_;
    static constexpr int singleCopyThreshold_ = 4000000;
};

} // namespace mirheo
