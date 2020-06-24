// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/utils/type_shift.h>
#include <mirheo/core/pvs/data_manager.h>

#include <mpi.h>
#include <tuple>
#include <vector>

namespace mirheo
{

class ParticleVector;
class ObjectVector;

namespace restart_helpers
{
constexpr int InvalidProc = -1;
constexpr int tag = 4243;

using VarVector = mpark::variant<
#define MAKE_WRAPPER(a) std::vector<a>
    MIRHEO_TYPE_TABLE_COMMA(MAKE_WRAPPER)
#undef MAKE_WRAPPER
    >;

/// Simple structure that describes a channel of data
struct NamedData
{
    std::string name; ///< channel name
    VarVector data;   ///< channel data
    bool needShift;   ///< channel shift mode
};

using ListData = std::vector<NamedData>;
using ExchMap  = std::vector<int>;

ListData readData(const std::string& filename, MPI_Comm comm, int chunkSize);

template<typename T>
std::vector<T> extractChannel(const std::string& name, ListData& channels)
{
    using VecType = std::vector<T>;

    for (auto it = channels.begin(); it != channels.end(); ++it)
    {
        if (it->name != name) continue;

        if (mpark::holds_alternative<VecType>(it->data))
        {
            VecType v {std::move(mpark::get<VecType>(it->data))};
            channels.erase(it);
            return v;
        }
        else
        {
            mpark::visit([&](const auto& vec)
            {
                using VectorType = typename std::remove_reference<decltype(vec)>::type::value_type;
                die ("could not retrieve channel '%s' with given type: got %s instead of %s",
                     name.c_str(), typeid(VectorType).name(), typeid(T).name());
            }, it->data);
        }
    }
    die ("could not find channel '%s'", name.c_str());
    return {};
}

ExchMap getExchangeMap(MPI_Comm comm, const DomainInfo domain,
                       int objSize, const std::vector<real3>& positions);

std::tuple<std::vector<real4>, std::vector<real4>>
combinePosVelIds(const std::vector<real3>& pos,
                 const std::vector<real3>& vel,
                 const std::vector<int64_t>& ids);

std::vector<RigidMotion>
combineMotions(const std::vector<real3>& pos,
               const std::vector<RigidReal4>& quaternion,
               const std::vector<RigidReal3>& vel,
               const std::vector<RigidReal3>& omega,
               const std::vector<RigidReal3>& force,
               const std::vector<RigidReal3>& torque);

namespace details
{
template <typename T>
static std::vector<std::vector<T>> splitData(const ExchMap& map, int chunkSize,
                                             const std::vector<T>& data, int numProcs)
{
    std::vector<std::vector<T>> bufs(numProcs);

    for (size_t i = 0; i < map.size(); ++i)
    {
        const int procId = map[i];

        if (procId == InvalidProc) continue;

        bufs[procId].insert(bufs[procId].end(),
                            data.begin() +  i      * chunkSize,
                            data.begin() + (i + 1) * chunkSize);
    }

    return bufs;
}

template<typename T>
static std::vector<MPI_Request> sendData(const std::vector<std::vector<T>>& sendBufs, MPI_Comm comm)
{
    std::vector<MPI_Request> reqs;

    for (size_t i = 0; i < sendBufs.size(); ++i)
    {
        MPI_Request req;
        debug3("Sending %zu elements to rank %zu", sendBufs[i].size(), i);
        MPI_Check( MPI_Isend(sendBufs[i].data(), static_cast<int>(sendBufs[i].size() * sizeof(T)),
                             MPI_BYTE, (int) i, tag, comm, &req) );
        reqs.push_back(req);
    }
    return reqs;
}

template <typename T>
static std::vector<T> recvData(int numProcs, MPI_Comm comm)
{
    std::vector<T> allData;
    for (int i = 0; i < numProcs; ++i)
    {
        MPI_Status status;
        int sizeBytes;
        std::vector<T> recvBuf;

        MPI_Check( MPI_Probe(i, tag, comm, &status) );
        MPI_Check( MPI_Get_count(&status, MPI_BYTE, &sizeBytes) );

        const int size = sizeBytes / (int) sizeof(T);

        if (static_cast<int>(size * sizeof(T)) != sizeBytes)
            die("unexpected received size: got %ld bytes, expected multiple of %ld",
                static_cast<long>(sizeBytes), static_cast<long>(sizeof(T)));

        recvBuf.resize(size);

        debug3("Receiving %d elements from %d", size, status.MPI_SOURCE);
        MPI_Check( MPI_Recv(recvBuf.data(), sizeBytes, MPI_BYTE,
                            status.MPI_SOURCE, tag, comm, MPI_STATUS_IGNORE) );

        allData.insert(allData.end(), recvBuf.begin(), recvBuf.end());
    }
    return allData;
}

inline int getNumProcs(MPI_Comm comm)
{
    int numProcs;
    MPI_Check( MPI_Comm_size(comm, &numProcs) );
    return numProcs;
}

} // namespace details

template<typename T>
static void exchangeData(MPI_Comm comm, const ExchMap& map,
                         std::vector<T>& data, int chunkSize = 1)
{
    const int numProcs = details::getNumProcs(comm);

    auto sendBufs = details::splitData(map, chunkSize, data, numProcs);
    auto sendReqs = details::sendData(sendBufs, comm);
    data          = details::recvData<T>(numProcs, comm);

    MPI_Check( MPI_Waitall((int) sendReqs.size(), sendReqs.data(), MPI_STATUSES_IGNORE) );
}

void exchangeListData(MPI_Comm comm, const ExchMap& map, ListData& listData, int chunkSize = 1);

template<typename Container>
static void shiftElementsGlobal2Local(Container& data, const DomainInfo domain)
{
    auto shift = domain.global2local({0._r, 0._r, 0._r});
    for (auto& d : data) type_shift::apply(d, shift);
}

void requireExtraDataPerParticle(const ListData& listData, ParticleVector *pv);
void requireExtraDataPerObject  (const ListData& listData, ObjectVector   *ov);

void copyAndShiftListData(const DomainInfo domain,
                          const ListData& listData,
                          DataManager& dataManager);

} // namespace restart_helpers

} // namespace mirheo
