#include "particle_exchanger.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

#include <algorithm>

ExchangeHelper::ExchangeHelper(std::string name, const int datumSize) :
    name(name), datumSize(datumSize)
{
    recvSizes.  resize_anew(nBuffers);
    recvOffsets.resize_anew(nBuffers+1);

    sendSizes.  resize_anew(nBuffers);
    sendOffsets.resize_anew(nBuffers+1);
}

void ExchangeHelper::makeOffsets(const PinnedBuffer<int>& sz, PinnedBuffer<int>& of)
{
    int n = sz.size();
    if (n == 0) return;

    of[0] = 0;
    for (int i=0; i < n; i++)
        of[i+1] = of[i] + sz[i];
}

ParticleExchanger::ParticleExchanger(MPI_Comm& comm, bool gpuAwareMPI) :
        nActiveNeighbours(26), gpuAwareMPI(gpuAwareMPI), singleCopyThreshold(1<<18)
{
    MPI_Check( MPI_Comm_dup(comm, &haloComm) );

    int dims[3], periods[3], coords[3];
    MPI_Check( MPI_Cart_get (haloComm, 3, dims, periods, coords) );
    MPI_Check( MPI_Comm_rank(haloComm, &myrank));

    for(int i = 0; i < 27; ++i)
    {
        int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

        int coordsNeigh[3];
        for(int c = 0; c < 3; ++c)
            coordsNeigh[c] = coords[c] + d[c];

        MPI_Check( MPI_Cart_rank(haloComm, coordsNeigh, dir2rank + i) );

        dir2sendTag[i] = i;

        int cx = -( i%3 - 1 ) + 1;
        int cy = -( (i/3)%3 - 1 ) + 1;
        int cz = -( i/9 - 1 ) + 1;
        dir2recvTag[i] = (cz*3 + cy)*3 + cx;
    }
}

void ParticleExchanger::init(cudaStream_t stream)
{
    // Post irecv for sizes
    for (int i=0; i<helpers.size(); i++)
        if (needExchange(i)) postRecvSize(helpers[i]);

    // Derived class determines what to send
    for (int i=0; i<helpers.size(); i++)
        if (needExchange(i)) prepareSizes(i, stream);

    // Send sizes
    for (int i=0; i<helpers.size(); i++)
        if (needExchange(i)) sendSizes(helpers[i]);

    // Derived class determines what to send
    for (int i=0; i<helpers.size(); i++)
        if (needExchange(i)) prepareData(i, stream);

    // Post big data irecv (after prepereData cause it waits for the sizes)
    for (int i=0; i<helpers.size(); i++)
        if (needExchange(i)) postRecv(helpers[i]);

    // Send
    for (int i=0; i<helpers.size(); i++)
        if (needExchange(i)) send(helpers[i], stream);
}

void ParticleExchanger::finalize(cudaStream_t stream)
{
    // Wait for the irecvs to finish
    for (int i=0; i<helpers.size(); i++)
        if (needExchange(i)) wait(helpers[i], stream);

    // Derived class unpack implementation
    for (int i=0; i<helpers.size(); i++)
        if (needExchange(i)) combineAndUploadData(i, stream);
}


int ParticleExchanger::tagByName(std::string name)
{
    // TODO: better tagging policy (unique id?)
    static std::hash<std::string> nameHash;
    return (int)( nameHash(name) % (32767 / 27) );
}


void ParticleExchanger::postRecvSize(ExchangeHelper* helper)
{
    std::string pvName = helper->name;

    auto nBuffers = helper->nBuffers;
    auto rSizes   = helper->recvSizes.  hostPtr();
    auto rOffsets = helper->recvOffsets.hostPtr();

    // Receive sizes
    helper->requests.clear();
    helper->recvSizes.clearHost();

    for (int i=0; i < nBuffers; i++)
        if (i != 13 && dir2rank[i] >= 0)
        {
            MPI_Request req;
            const int tag = nBuffers * tagByName(pvName) + dir2recvTag[i];

            MPI_Check( MPI_Irecv(rSizes + i, 1, MPI_INT, dir2rank[i], tag, haloComm, &req) );
            helper->requests.push_back(req);
        }
}

/**
 * Expects helper->sendSizes and helper->sendOffsets to be ON HOST
 */
void ParticleExchanger::sendSizes(ExchangeHelper* helper)
{
    std::string pvName = helper->name;

    auto nBuffers = helper->nBuffers;
    auto sSizes   = helper->sendSizes.hostPtr();

    // Do blocking send in hope that it will be immediate due to small size
    for (int i=0; i < nBuffers; i++)
        if (i != 13 && dir2rank[i] >= 0)
        {
            const int tag = nBuffers * tagByName(pvName) + dir2sendTag[i];
            MPI_Check( MPI_Send(sSizes+i, 1, MPI_INT, dir2rank[i], tag, haloComm) );
        }
}

void ParticleExchanger::postRecv(ExchangeHelper* helper)
{
    std::string pvName = helper->name;

    auto nBuffers = helper->nBuffers;
    auto rSizes   = helper->recvSizes.  hostPtr();
    auto rOffsets = helper->recvOffsets.hostPtr();

    MPI_Check( MPI_Waitall(helper->requests.size(), helper->requests.data(), MPI_STATUSES_IGNORE) );

    // Prepare offsets and resize
    helper->makeRecvOffsets();
    int totalRecvd = rOffsets[nBuffers];
    helper->resizeRecvBuf();

    // Now do the actual data recv
    helper->requests.clear();
    helper->reqIndex.clear();
    for (int i=0; i < nBuffers; i++)
        if (i != 13 && dir2rank[i] >= 0)
        {
            MPI_Request req;
            const int tag = nBuffers * tagByName(pvName) + dir2recvTag[i];

            debug3("Receiving %s entities from rank %d, %d entities (buffer %d)",
                    pvName.c_str(), dir2rank[i], rSizes[i], i);

            if (rSizes[i] > 0)
            {
                auto ptr = gpuAwareMPI ? helper->recvBuf.devPtr() : helper->recvBuf.hostPtr();
                
                MPI_Check( MPI_Irecv(
                        ptr + rOffsets[i]*helper->datumSize,
                        rSizes[i]*helper->datumSize,
                        MPI_BYTE, dir2rank[i], tag, haloComm, &req) );

                helper->requests.push_back(req);
                helper->reqIndex.push_back(i);
            }
        }

    debug("Posted receive for %d %s entities", totalRecvd, pvName.c_str());
}

/**
 * helper->recvBuf will contain all the data, ON DEVICE already
 */
void ParticleExchanger::wait(ExchangeHelper* helper, cudaStream_t stream)
{
    auto rSizes   = helper->recvSizes.  hostPtr();
    auto rOffsets = helper->recvOffsets.hostPtr();
    bool singleCopy = helper->sendBuf.size() < singleCopyThreshold;
    
    // Wait for all if we want to copy all at once
    if (singleCopy || gpuAwareMPI)
    {
        MPI_Check( MPI_Waitall(helper->requests.size(), helper->requests.data(), MPI_STATUSES_IGNORE) );
        if (!gpuAwareMPI)
            helper->recvBuf.uploadToDevice(stream);
    }
    else
    {
        // Wait and upload one by one
        for (int i=0; i<helper->requests.size(); i++)    
        {
            int idx;
            MPI_Check( MPI_Waitany(helper->requests.size(), helper->requests.data(), &idx, MPI_STATUS_IGNORE) );
            int from = helper->reqIndex[idx];

            CUDA_Check( cudaMemcpyAsync(
                            helper->recvBuf.devPtr()  + rOffsets[from]*helper->datumSize,
                            helper->recvBuf.hostPtr() + rOffsets[from]*helper->datumSize,
                            rSizes[from] * helper->datumSize,
                            cudaMemcpyHostToDevice, stream) );
        }
    }

    // And report!
    debug("Completed receive for %s", helper->name.c_str());
}

/**
 * Expects helper->sendSizes and helper->sendOffsets to be ON HOST
 * helper->sendBuf data is ON DEVICE
 */
void ParticleExchanger::send(ExchangeHelper* helper, cudaStream_t stream)
{
    std::string pvName = helper->name;

    auto nBuffers = helper->nBuffers;
    auto sSizes   = helper->sendSizes.  hostPtr();
    auto sOffsets = helper->sendOffsets.hostPtr();
    bool singleCopy = helper->sendBuf.size() < singleCopyThreshold;

    if (!gpuAwareMPI && singleCopy)
        helper->sendBuf.downloadFromDevice(stream);

    MPI_Request req;
    int totSent = 0;
    for (int i=0; i < nBuffers; i++)
        if (i != 13 && dir2rank[i] >= 0)
        {
            debug3("Sending %s entities to rank %d in dircode %d [%2d %2d %2d], %d entities",
                    pvName.c_str(), dir2rank[i], i, i%3 - 1, (i/3)%3 - 1, i/9 - 1, sSizes[i]);

            const int tag = nBuffers * tagByName(pvName) + dir2sendTag[i];

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
                MPI_Check( MPI_Request_free(&req) );
            }

            totSent += sSizes[i];
        }

    debug("Sent total %d %s entities", totSent, pvName.c_str());
}


