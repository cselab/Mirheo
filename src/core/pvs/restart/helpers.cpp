#include "helpers.h"

#include <core/rigid_kernels/rigid_motion.h>
#include <core/utils/cuda_common.h>
#include <core/xdmf/xdmf.h>

#include <numeric>

namespace RestartHelpers
{

namespace details
{
struct getVarTypeVisitor
{
    int ncomp;

    template<typename T>
    VarVector operator()(DataTypeWrapper<T>) const {return std::vector<T>{};}

    VarVector operator()(DataTypeWrapper<float>) const
    {
        switch(ncomp)
        {
        case 2:  return std::vector<float2>{}; break;
        case 3:  return std::vector<float3>{}; break;
        case 4:  return std::vector<float4>{}; break;
        default: return std::vector<float>{}; break;
        }
    }

    VarVector operator()(DataTypeWrapper<double>) const
    {
        switch(ncomp)
        {
        case 3:  return std::vector<double3>{}; break;
        case 4:  return std::vector<double4>{}; break;
        default: return std::vector<double4>{}; break;
        }
    }
};
} // namespace details

ListData readData(const std::string& filename, MPI_Comm comm, int chunkSize)
{
    auto vertexData = XDMF::readVertexData(filename, comm, chunkSize);
    const size_t n = vertexData.positions.size();

    ListData listData {{ChannelNames::XDMF::position, vertexData.positions}};

    for (const auto& desc : vertexData.descriptions)
    {
        int ncomp   = XDMF::dataFormToNcomponents(desc.dataForm);
        auto varVec = mpark::visit(details::getVarTypeVisitor{ncomp}, desc.type);
        
        mpark::visit([&](auto& dstVec)
        {
            using T = typename std::remove_reference<decltype(dstVec)>::type::value_type;
            auto srcData = reinterpret_cast<const T*>(desc.data);
            NamedData nd {desc.name, std::vector<T>{srcData, srcData + n}};
            listData.push_back(std::move(nd));
        }, varVec);
    }
    return listData;
}

static ExchMap getExchangeMapFromPos(MPI_Comm comm, const DomainInfo domain,
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

ExchMap getExchangeMap(MPI_Comm comm, const DomainInfo domain,
                       int objSize, const std::vector<float3>& positions)
{
    int nObjs = positions.size() / objSize;

    if (positions.size() % objSize != 0)
        die("expected a multiple of %d, got %d", objSize, (int)positions.size());

    std::vector<float3> coms;
    coms.reserve(nObjs);

    constexpr float3 zero3 {0.f, 0.f, 0.f};
    const float factor = 1.0 / objSize;
    
    for (int i = 0; i < nObjs; ++i)
    {
        float3 com = factor * std::accumulate(positions.data() + (i + 0) * objSize,
                                              positions.data() + (i + 1) * objSize,
                                              zero3);
        coms.push_back(com);
    }

    return getExchangeMapFromPos(comm, domain, coms);
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
        p.u = vel[i];
        p.setId(ids[i]);

        pos4[i] = p.r2Float4();
        vel4[i] = p.u2Float4();
    }
    return {pos4, vel4};
}

std::vector<RigidMotion>
combineMotions(const std::vector<float3>& pos,
               const std::vector<RigidReal4>& quaternion,
               const std::vector<RigidReal3>& vel,
               const std::vector<RigidReal3>& omega,
               const std::vector<RigidReal3>& force,
               const std::vector<RigidReal3>& torque)
{
    auto n = pos.size();
    std::vector<RigidMotion> motions(n);

    for (size_t i = 0; i < n; ++i)
    {
        RigidMotion m;
        m.r      = make_rigidReal3(pos[i]);
        m.q      = quaternion[i];
        m.vel    = vel       [i];
        m.omega  = omega     [i];
        m.force  = force     [i];
        m.torque = torque    [i];
        motions[i] = m;
    }
    return motions;
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

void exchangeListData(MPI_Comm comm, const ExchMap& map, ListData& listData, int chunkSize)
{
    for (auto& entry : listData)
    {
        mpark::visit([&](auto& data)
        {
            exchangeData(comm, map, data, chunkSize);
        }, entry.data);
    }
}

int getLocalNumElementsAfterExchange(MPI_Comm comm, const ExchMap& map)
{
    int numProcs, procId;
    MPI_Check( MPI_Comm_rank(comm, &procId) );
    MPI_Check( MPI_Comm_size(comm, &numProcs) );

    std::vector<int> numElements(numProcs, 0);
    for (auto pid : map)
        numElements[pid]++;

    MPI_Check( MPI_Allreduce(MPI_IN_PLACE, numElements.data(), numElements.size(),
                             MPI_INT, MPI_SUM, comm) );

    return numElements[procId];
}

} // namespace RestartHelpers
