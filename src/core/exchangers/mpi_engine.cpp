#include "mpi_engine.h"
#include "utils/fragments_mapping.h"

#include <core/utils/timer.h>
#include <core/logger.h>
#include <algorithm>

MPIExchangeEngine::MPIExchangeEngine(std::unique_ptr<Exchanger> exchanger,
                                     MPI_Comm comm, bool gpuAwareMPI) :
    nActiveNeighbours(FragmentMapping::numFragments - 1),
    gpuAwareMPI(gpuAwareMPI),
    exchanger(std::move(exchanger))
{
    MPI_Check( MPI_Comm_dup(comm, &haloComm) );

    int dims[3], periods[3], coords[3];
    MPI_Check( MPI_Cart_get (haloComm, 3, dims, periods, coords) );
    MPI_Check( MPI_Comm_rank(haloComm, &myrank));

    for (int i = 0; i < FragmentMapping::numFragments; ++i)
    {
        int d[3] = { FragmentMapping::getDirx(i),
                     FragmentMapping::getDiry(i),
                     FragmentMapping::getDirz(i) };

        int coordsNeigh[3];
        for(int c = 0; c < 3; ++c)
            coordsNeigh[c] = coords[c] + d[c];

        MPI_Check( MPI_Cart_rank(haloComm, coordsNeigh, dir2rank + i) );

        dir2sendTag[i] = i;
        dir2recvTag[i] = FragmentMapping::getId(-d[0], -d[1], -d[2]);
    }
}

MPIExchangeEngine::~MPIExchangeEngine()
{
    MPI_Check( MPI_Comm_free(&haloComm) );
}

void MPIExchangeEngine::init(cudaStream_t stream)
{
    auto& helpers = exchanger->helpers;
    
    for (int i = 0; i < helpers.size(); i++)
        if (!exchanger->needExchange(i)) debug("Exchange of PV '%s' is skipped", helpers[i]->name.c_str());
    
    // Post irecv for sizes
    for (int i = 0; i < helpers.size(); i++)
        if (exchanger->needExchange(i)) postRecvSize(helpers[i].get());

    // Derived class determines what to send
    for (int i = 0; i < helpers.size(); i++)
        if (exchanger->needExchange(i)) exchanger->prepareSizes(i, stream);

    // Send sizes
    for (int i = 0; i < helpers.size(); i++)
        if (exchanger->needExchange(i)) sendSizes(helpers[i].get());

    // Derived class determines what to send
    for (int i = 0; i < helpers.size(); i++)
        if (exchanger->needExchange(i)) exchanger->prepareData(i, stream);

    // Post big data irecv (after prepereData cause it waits for the sizes)
    for (int i = 0; i < helpers.size(); i++)
        if (exchanger->needExchange(i)) postRecv(helpers[i].get());

    // CUDA-aware MPI will work in a separate stream, need to synchro
    if (gpuAwareMPI) cudaStreamSynchronize(stream);

    // Send
    for (int i = 0; i < helpers.size(); i++)
        if (exchanger->needExchange(i)) send(helpers[i].get(), stream);
}

void MPIExchangeEngine::finalize(cudaStream_t stream)
{
    auto& helpers = exchanger->helpers;

    // Wait for the irecvs to finish
    for (int i = 0; i < helpers.size(); i++)
        if (exchanger->needExchange(i)) wait(helpers[i].get(), stream);

    // Wait for completion of the previous sends
	for (int i = 0; i < helpers.size(); i++)
		if (exchanger->needExchange(i))
			MPI_Check( MPI_Waitall(
					helpers[i]->sendRequests.size(),
					helpers[i]->sendRequests.data(),
					MPI_STATUSES_IGNORE) );

    // Derived class unpack implementation
    for (int i = 0; i < helpers.size(); i++)
        if (exchanger->needExchange(i)) exchanger->combineAndUploadData(i, stream);
}

void MPIExchangeEngine::postRecvSize(ExchangeHelper* helper)
{
    std::string pvName = helper->name;

    auto nBuffers = helper->nBuffers;
    auto bulkId   = helper->bulkId;
    auto rSizes   = helper->recvSizes.  hostPtr();
    auto rOffsets = helper->recvOffsets.hostPtr();

    // Receive sizes
    helper->recvRequests.clear();
    helper->recvSizes.clearHost();

    for (int i = 0; i < nBuffers; i++)
        if (i != bulkId && dir2rank[i] >= 0)
        {
            MPI_Request req;
            const int tag = nBuffers * helper->getUniqueId() + dir2recvTag[i];

            MPI_Check( MPI_Irecv(rSizes + i, 1, MPI_INT, dir2rank[i], tag, haloComm, &req) );
            helper->recvRequests.push_back(req);
        }
}

/**
 * Expects helper->sendSizes and helper->sendOffsets to be ON HOST
 */
void MPIExchangeEngine::sendSizes(ExchangeHelper* helper)
{
    std::string pvName = helper->name;

    auto nBuffers = helper->nBuffers;
    auto bulkId   = helper->bulkId;
    auto sSizes   = helper->sendSizes.hostPtr();

    // Do blocking send in hope that it will be immediate due to small size
    for (int i = 0; i < nBuffers; i++)
        if (i != bulkId && dir2rank[i] >= 0)
        {
            const int tag = nBuffers * helper->getUniqueId() + dir2sendTag[i];
            MPI_Check( MPI_Send(sSizes+i, 1, MPI_INT, dir2rank[i], tag, haloComm) );
        }
}

static void safeWaitAll(int count, MPI_Request array_of_requests[])
{
    std::vector<MPI_Status> statuses(count);
    int code = MPI_Waitall(count, array_of_requests, statuses.data());

    if (code != MPI_SUCCESS) {
        std::string allErrors;
        char buf[MPI_MAX_ERROR_STRING];
        int nchar;
        MPI_Error_string(code, buf, &nchar);
        allErrors += std::string(buf) + "\n";
        for (int i = 0; i < count; ++i) {            
            MPI_Error_string(statuses[i].MPI_ERROR, buf, &nchar);
            allErrors += std::string(buf) + "\n";
        }

        die("Waitall errors:\n%s", allErrors.c_str());
    }
}

void MPIExchangeEngine::postRecv(ExchangeHelper* helper)
{
    std::string pvName = helper->name;

    auto nBuffers = helper->nBuffers;
    auto bulkId   = helper->bulkId;
    auto rSizes   = helper->recvSizes.  hostPtr();
    auto rOffsets = helper->recvOffsets.hostPtr();

    mTimer tm;
    tm.start();
    // MPI_Check( MPI_Waitall(helper->recvRequests.size(), helper->recvRequests.data(), MPI_STATUSES_IGNORE) );
    safeWaitAll(helper->recvRequests.size(), helper->recvRequests.data());
    debug("Waiting for sizes of '%s' took %f ms", pvName.c_str(), tm.elapsed());

    // Prepare offsets and resize
    helper->computeRecvOffsets();
    int totalRecvd = rOffsets[nBuffers];
    helper->resizeRecvBuf();

    // Now do the actual data recv
    helper->recvRequests.clear();
    helper->recvRequestIdxs.clear();
    for (int i = 0; i < nBuffers; i++)
        if (i != bulkId && dir2rank[i] >= 0)
        {
            MPI_Request req;
            const int tag = nBuffers * helper->getUniqueId() + dir2recvTag[i];

            debug3("Receiving %s entities from rank %d, %d entities (buffer %d, datum size %d)",
                   pvName.c_str(), dir2rank[i], rSizes[i], i, helper->datumSize);

            if (rSizes[i] > 0)
            {
                auto ptr = gpuAwareMPI ? helper->recvBuf.devPtr() : helper->recvBuf.hostPtr();
                
                MPI_Check( MPI_Irecv(
                        ptr + rOffsets[i]*helper->datumSize,
                        rSizes[i]*helper->datumSize,
                        MPI_BYTE, dir2rank[i], tag, haloComm, &req) );

                helper->recvRequests.push_back(req);
                helper->recvRequestIdxs.push_back(i);
            }
        }

    debug("Posted receive for %d %s entities", totalRecvd, pvName.c_str());
}

/**
 * helper->recvBuf will contain all the data, ON DEVICE already
 */
void MPIExchangeEngine::wait(ExchangeHelper* helper, cudaStream_t stream)
{
    std::string pvName = helper->name;

    auto rSizes   = helper->recvSizes.  hostPtr();
    auto rOffsets = helper->recvOffsets.hostPtr();
    bool singleCopy = helper->recvOffsets[FragmentMapping::numFragments] * helper->datumSize < singleCopyThreshold;
    
    debug("Waiting to receive '%s' entities, single copy is %s, GPU aware MPI is %s",
        pvName.c_str(), singleCopy ? "on" : "off", gpuAwareMPI ? "on" : "off");
    
    double waitTime = 0;
    mTimer tm;
    // Wait for all if we want to copy all at once
    if (singleCopy || gpuAwareMPI)
    {
        tm.start();
        // MPI_Check( MPI_Waitall(helper->recvRequests.size(), helper->recvRequests.data(), MPI_STATUSES_IGNORE) );
        safeWaitAll(helper->recvRequests.size(), helper->recvRequests.data());
        waitTime = tm.elapsed();
        if (!gpuAwareMPI)
            helper->recvBuf.uploadToDevice(stream);
    }
    else
    {
        // Wait and upload one by one
        for (int i = 0; i < helper->recvRequests.size(); i++)
        {
            int idx;
            tm.start();
            MPI_Check( MPI_Waitany(helper->recvRequests.size(), helper->recvRequests.data(), &idx, MPI_STATUS_IGNORE) );
            waitTime += tm.elapsedAndReset();

            int from = helper->recvRequestIdxs[idx];

            CUDA_Check( cudaMemcpyAsync(
                            helper->recvBuf.devPtr()  + rOffsets[from]*helper->datumSize,
                            helper->recvBuf.hostPtr() + rOffsets[from]*helper->datumSize,
                            rSizes[from] * helper->datumSize,
                            cudaMemcpyHostToDevice, stream) );
        }
    }

    // And report!
    debug("Completed receive for '%s', waiting took %f ms", helper->name.c_str(), waitTime);
}

/**
 * Expects helper->sendSizes and helper->sendOffsets to be ON HOST
 * helper->sendBuf data is ON DEVICE
 */
void MPIExchangeEngine::send(ExchangeHelper* helper, cudaStream_t stream)
{
    std::string pvName = helper->name;

    auto nBuffers = helper->nBuffers;
    auto bulkId   = helper->bulkId;
    auto sSizes   = helper->sendSizes.  hostPtr();
    auto sOffsets = helper->sendOffsets.hostPtr();
    bool singleCopy = helper->sendBuf.size() < singleCopyThreshold;
    
    debug("Sending '%s' entities, single copy is %s, GPU aware MPI is %s",
        pvName.c_str(), singleCopy ? "on" : "off", gpuAwareMPI ? "on" : "off");

    if (!gpuAwareMPI && singleCopy)
        helper->sendBuf.downloadFromDevice(stream);

    MPI_Request req;
    int totSent = 0;
    helper->sendRequests.clear();

    for (int i = 0; i < nBuffers; i++)
        if (i != bulkId && dir2rank[i] >= 0)
        {
            debug3("Sending %s entities to rank %d in dircode %d [%2d %2d %2d], %d entities",
                   pvName.c_str(), dir2rank[i], i, FragmentMapping::getDirx(i), FragmentMapping::getDiry(i), FragmentMapping::getDirz(i), sSizes[i]);

            const int tag = nBuffers * helper->getUniqueId() + dir2sendTag[i];

            // Send actual data
            if (sSizes[i] > 0)
            {
                auto ptr = gpuAwareMPI ? helper->sendBuf.devPtr() : helper->sendBuf.hostPtr();
                
                if (!singleCopy && (!gpuAwareMPI))
                {
                    CUDA_Check( cudaMemcpyAsync(
                                    helper->sendBuf.hostPtr() + sOffsets[i]*helper->datumSize,
                                    helper->sendBuf.devPtr()  + sOffsets[i]*helper->datumSize,
                                    sSizes[i] * helper->datumSize,
                                    cudaMemcpyDeviceToHost, stream) );
                    CUDA_Check( cudaStreamSynchronize(stream) );
                }

                MPI_Check( MPI_Isend(
                        ptr + sOffsets[i]*helper->datumSize,
                        sSizes[i] * helper->datumSize,
                        MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
                helper->sendRequests.push_back(req);
            }

            totSent += sSizes[i];
        }

    debug("Sent total %d '%s' entities", totSent, pvName.c_str());
}


