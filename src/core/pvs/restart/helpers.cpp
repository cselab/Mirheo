#include "helpers.h"

#include <core/utils/cuda_common.h>
#include <core/xdmf/xdmf.h>

namespace RestartHelpers
{

ListData readData(const std::string& filename, MPI_Comm comm, int chunkSize)
{
    auto vertexData = XDMF::readVertexData(filename, comm, chunkSize);
    const size_t n = vertexData.positions->size();

    ListData listData {{ChannelNames::XDMF::position, *vertexData.positions}};

    for (const auto& desc : vertexData.descriptions)
    {
        mpark::visit([&](auto typeWrapper)
        {
            using T = typename decltype(typeWrapper)::type;
            auto dataPtr = reinterpret_cast<const T*>(desc.data);

            NamedData nd {desc.name, std::vector<T>{dataPtr, dataPtr + n}};
            listData.push_back(std::move(nd));
        }, desc.type);
    }
    return listData;
}

ExchMap getExchangeMap(MPI_Comm comm, const DomainInfo domain,
                       const std::vector<float3>& positions)
{
    int dims[3], periods[3], coords[3];
    MPI_Check( MPI_Cart_get(comm, 3, dims, periods, coords) );

    ExchMap map;
    map.reserve(positions.size());
    int numberInvalid = 0;
    
    for (auto r : positions)
    {
        int3 procId3 = make_int3(floorf(r / domain.localSize));

        if (procId3.x >= dims[0] ||
            procId3.y >= dims[1] ||
            procId3.z >= dims[2])
        {
            map.push_back(InvalidProc);
            ++ numberInvalid;
            continue;
        }
        else
        {
            int procId;
            MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
            map.push_back(procId);
        }
    }

    if (numberInvalid)
        warn("Restart: skipped %d invalid particle positions", numberInvalid);

    return map;    
}

std::tuple<std::vector<float4>, std::vector<float4>>
combinePosVelIds(const std::vector<float3>& pos,
                 const std::vector<float3>& vel,
                 const std::vector<int64_t>& ids)
{
    auto n = pos.size();
    std::vector<float4> pos4(n), vel4(n);

    for (size_t i = 0; i < n; ++i)
    {
        Particle p;
        p.r = pos[i];
        p.u = pos[i];
        p.setId(ids[i]);

        pos4[i] = p.r2Float4();
        vel4[i] = p.u2Float4();
    }
    return {pos4, vel4};
}


void copyShiftCoordinates(const DomainInfo &domain, const std::vector<float4>& pos, const std::vector<float4>& vel,
                          LocalParticleVector *local)
{
    auto& positions  = local->positions();
    auto& velocities = local->velocities();

    positions .resize(pos.size(), defaultStream);
    velocities.resize(vel.size(), defaultStream);
    
    for (int i = 0; i < pos.size(); i++) {
        auto p = Particle(pos[i], vel[i]);
        p.r = domain.global2local(p.r);
        positions [i] = p.r2Float4();
        velocities[i] = p.u2Float4();
    }
}

} // namespace RestartHelpers
