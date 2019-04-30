#pragma once

#include <mpi.h>
#include <vector>

#include "core/domain.h"
#include "particle_vector.h"

namespace RestartHelpers
{
void copyShiftCoordinates(const DomainInfo &domain, const std::vector<float4>& pos, const std::vector<float4>& vel,
                          LocalParticleVector *local);

template<typename T>
static void sendData(const std::vector<std::vector<T>> &sendBufs, std::vector<MPI_Request> &reqs, MPI_Comm comm)
{
    for (int i = 0; i < sendBufs.size(); i++) {
        debug3("Sending %d elements to rank %d", sendBufs[i].size(), i);
        MPI_Check( MPI_Isend(sendBufs[i].data(), sendBufs[i].size() * sizeof(T), MPI_BYTE, i, 0, comm, reqs.data()+i) );
    }
}

template <typename T>
static void recvData(int size, std::vector<T> &all, MPI_Comm comm)
{
    all.resize(0);
    for (int i = 0; i < size; i++) {
        MPI_Status status;
        int msize;
        std::vector<T> recvBuf;
        
        MPI_Check( MPI_Probe(MPI_ANY_SOURCE, 0, comm, &status) );
        MPI_Check( MPI_Get_count(&status, MPI_BYTE, &msize) );

        recvBuf.resize(msize / sizeof(T));

        debug3("Receiving %d elements from %d", msize, status.MPI_SOURCE);
        MPI_Check( MPI_Recv(recvBuf.data(), msize, MPI_BYTE, status.MPI_SOURCE, 0, comm, MPI_STATUS_IGNORE) );

        all.insert(all.end(), recvBuf.begin(), recvBuf.end());
    }
}

template <typename T>
static void splitData(const std::vector<int>& map, int chunkSize, const std::vector<T>& data, std::vector<std::vector<T>>& buffs)
{
    for (int i = 0; i < map.size(); ++i) {
        int procId = map[i];

        if (procId == -1) continue;            

        buffs[procId].insert(buffs[procId].end(),
                             data.begin() +  i      * chunkSize,
                             data.begin() + (i + 1) * chunkSize);
    }
}

template<typename T>
static void exchangeData(MPI_Comm comm, const std::vector<int>& map, std::vector<T>& data, int chunkSize = 1)
{
    int size;
    MPI_Check( MPI_Comm_size(comm, &size) );
    
    std::vector<std::vector<T>> sendBufs(size);
    std::vector<MPI_Request> reqs(size);
        
    splitData(map, chunkSize, data, sendBufs);
    sendData(sendBufs, reqs, comm);
    recvData(size, data, comm);

    MPI_Check( MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE) );
}

} // namespace RestartHelpers
