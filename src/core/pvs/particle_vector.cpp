#include "particle_vector.h"
#include "restart_helpers.h"

#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/xdmf/type_map.h>
#include <core/xdmf/xdmf.h>

#include <mpi.h>

LocalParticleVector::LocalParticleVector(ParticleVector *pv, int n) :
    pv(pv)
{
    dataPerParticle.createData<float4>(ChannelNames::positions,  n);
    dataPerParticle.createData<float4>(ChannelNames::velocities, n);
    dataPerParticle.createData<Force>(ChannelNames::forces, n);

    // positions are treated specially, do not need to be persistent
    dataPerParticle.setPersistenceMode(ChannelNames::velocities, DataManager::PersistenceMode::Persistent);
    resize_anew(n);
}

LocalParticleVector::~LocalParticleVector() = default;

void LocalParticleVector::resize(int n, cudaStream_t stream)
{
    if (n < 0) die("Tried to resize PV to %d < 0 particles", n);
    dataPerParticle.resize(n, stream);
    np = n;
}

void LocalParticleVector::resize_anew(int n)
{
    if (n < 0) die("Tried to resize PV to %d < 0 particles", n);
    dataPerParticle.resize_anew(n);
    np = n;
}

PinnedBuffer<float4>& LocalParticleVector::positions()
{
    return * dataPerParticle.getData<float4>(ChannelNames::positions);
}

PinnedBuffer<float4>& LocalParticleVector::velocities()
{
    return * dataPerParticle.getData<float4>(ChannelNames::velocities);
}

PinnedBuffer<Force>& LocalParticleVector::forces()
{
    return * dataPerParticle.getData<Force>(ChannelNames::forces);
}

void LocalParticleVector::computeGlobalIds(MPI_Comm comm, cudaStream_t stream)
{
    int64_t rankStart = 0;
    int64_t np64 = np;
        
    MPI_Check( MPI_Exscan(&np64, &rankStart, 1, MPI_INT64_T, MPI_SUM, comm) );

    auto& pos = positions();
    auto& vel = velocities();
    
    pos.downloadFromDevice(stream, ContainersSynch::Asynch);
    vel.downloadFromDevice(stream);

    int64_t id = rankStart;
    for (int i = 0; i < pos.size(); ++i)
    {
        Particle p(pos[i], vel[i]);
        p.setId(id++);
        pos[i] = p.r2Float4();
        vel[i] = p.u2Float4();
    }
    
    pos.uploadToDevice(stream);
    vel.uploadToDevice(stream);
}


//============================================================================
// Particle Vector
//============================================================================

ParticleVector::ParticleVector(const YmrState *state, std::string name, float mass, int n) :
    ParticleVector(state, name, mass,
                   std::make_unique<LocalParticleVector>(this, n),
                   std::make_unique<LocalParticleVector>(this, 0) )
{}

ParticleVector::ParticleVector(const YmrState *state, std::string name,  float mass,
                               std::unique_ptr<LocalParticleVector>&& local,
                               std::unique_ptr<LocalParticleVector>&& halo) :
    YmrSimulationObject(state, name),
    mass(mass),
    _local(std::move(local)),
    _halo(std::move(halo))
{
    // old positions and velocities don't need to exchanged in general
    requireDataPerParticle<float4> (ChannelNames::oldPositions, DataManager::PersistenceMode::None);
}

ParticleVector::~ParticleVector() = default;

std::vector<int64_t> ParticleVector::getIndices_vector()
{
    auto& pos = local()->positions();
    auto& vel = local()->velocities();
    pos.downloadFromDevice(defaultStream, ContainersSynch::Asynch);
    vel.downloadFromDevice(defaultStream);
    
    std::vector<int64_t> res(pos.size());

    for (size_t i = 0; i < pos.size(); i++)
    {
        Particle p (pos[i], vel[i]);
        res[i] = p.getId();
    }
    
    return res;
}

PyTypes::VectorOfFloat3 ParticleVector::getCoordinates_vector()
{
    auto& pos = local()->positions();
    pos.downloadFromDevice(defaultStream);
    
    PyTypes::VectorOfFloat3 res(pos.size());
    for (int i = 0; i < pos.size(); i++)
    {
        float3 r = make_float3(pos[i]);
        r = state->domain.local2global(r);
        res[i] = { r.x, r.y, r.z };
    }
    
    return res;
}

PyTypes::VectorOfFloat3 ParticleVector::getVelocities_vector()
{
    auto& vel = local()->velocities();
    vel.downloadFromDevice(defaultStream);
    
    PyTypes::VectorOfFloat3 res(vel.size());
    for (int i = 0; i < vel.size(); i++)
    {
        float3 u = make_float3(vel[i]);
        res[i] = { u.x, u.y, u.z };
    }
    
    return res;
}

PyTypes::VectorOfFloat3 ParticleVector::getForces_vector()
{
    HostBuffer<Force> forces;
    forces.copy(local()->forces(), defaultStream);
    
    PyTypes::VectorOfFloat3 res(forces.size());
    for (int i = 0; i < forces.size(); i++)
    {
        float3 f = forces[i].f;
        res[i] = { f.x, f.y, f.z };
    }
    
    return res;
}

void ParticleVector::setCoosVels_globally(PyTypes::VectorOfFloat6& coosvels, cudaStream_t stream)
{
    error("Not implemented yet");
}

void ParticleVector::setCoordinates_vector(PyTypes::VectorOfFloat3& coordinates)
{
    auto& pos = local()->positions();
    
    if (coordinates.size() != local()->size())
        throw std::invalid_argument("Wrong number of particles passed, "
            "expected: " + std::to_string(local()->size()) +
            ", got: " + std::to_string(coordinates.size()) );
    
    for (int i = 0; i < coordinates.size(); i++)
    {
        auto& r_ = coordinates[i];
        float3 r = state->domain.global2local( { r_[0], r_[1], r_[2] } );
        pos[i].x = r.x;
        pos[i].y = r.y;
        pos[i].z = r.z;
    }
    
    pos.uploadToDevice(defaultStream);
}

void ParticleVector::setVelocities_vector(PyTypes::VectorOfFloat3& velocities)
{
    auto& vel = local()->velocities();
    
    if (velocities.size() != local()->size())
        throw std::invalid_argument("Wrong number of particles passed, "
        "expected: " + std::to_string(local()->size()) +
        ", got: " + std::to_string(velocities.size()) );
    
    for (int i = 0; i < velocities.size(); i++)
    {
        auto& u = velocities[i];
        vel[i].x = u[0];
        vel[i].y = u[1];
        vel[i].z = u[2];
    }
    
    vel.uploadToDevice(defaultStream);
}

void ParticleVector::setForces_vector(PyTypes::VectorOfFloat3& forces)
{
    HostBuffer<Force> myforces(local()->size());
    
    if (forces.size() != local()->size())
        throw std::invalid_argument("Wrong number of particles passed, "
        "expected: " + std::to_string(local()->size()) +
        ", got: " + std::to_string(forces.size()) );
    
    for (int i = 0; i < forces.size(); i++)
    {
        auto& f = forces[i];
        myforces[i].f = { f[0], f[1], f[2] };
    }
    
    local()->forces().copy(myforces);
    local()->forces().uploadToDevice(defaultStream);
}

static void splitPV(DomainInfo domain, LocalParticleVector *local,
                    std::vector<float> &positions, std::vector<float> &velocities, std::vector<int64_t> &ids)
{
    int n = local->size();
    positions .resize(3 * n);
    velocities.resize(3 * n);
    ids.resize(n);

    auto pos4 = local->positions();
    auto vel4 = local->velocities();
    
    float3 *pos = (float3*) positions.data(), *vel = (float3*) velocities.data();
    
    for (int i = 0; i < n; i++)
    {
        auto p = Particle(pos4[i], vel4[i]);
        pos[i] = domain.local2global(p.r);
        vel[i] = p.u;
        ids[i] = p.getId();
    }
}

void ParticleVector::_extractPersistentExtraData(DataManager& extraData, std::vector<XDMF::Channel>& channels,
                                                 const std::set<std::string>& blackList)
{
    for (auto& namedChannelDesc : extraData.getSortedChannels())
    {        
        auto channelName = namedChannelDesc.first;
        auto channelDesc = namedChannelDesc.second;        
        
        if (channelDesc->persistence != DataManager::PersistenceMode::Persistent)
            continue;

        if (blackList.find(channelName) != blackList.end())
            continue;

        mpark::visit([&](auto bufferPtr) {
                         using T = typename std::remove_pointer<decltype(bufferPtr)>::type::value_type;
                         bufferPtr->downloadFromDevice(defaultStream, ContainersSynch::Synch);
                         auto formtype   = XDMF::getDataForm<T>();
                         auto numbertype = XDMF::getNumberType<T>();
                         auto datatype   = DataTypeWrapper<T>();
                         channels.push_back(XDMF::Channel(channelName, bufferPtr->data(), formtype, numbertype, datatype )); \
                     }, channelDesc->varDataPtr);
    }
}

void ParticleVector::_extractPersistentExtraParticleData(std::vector<XDMF::Channel>& channels, const std::set<std::string>& blackList)
{
    _extractPersistentExtraData(local()->dataPerParticle, channels, blackList);
}

void ParticleVector::_checkpointParticleData(MPI_Comm comm, std::string path, int checkpointId)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointNameWithId(path, "PV", "", checkpointId);
    info("Checkpoint for particle vector '%s', writing to file %s", name.c_str(), filename.c_str());

    local()->positions ().downloadFromDevice(defaultStream, ContainersSynch::Asynch);
    local()->velocities().downloadFromDevice(defaultStream, ContainersSynch::Synch);

    auto positions = std::make_shared<std::vector<float>>();
    std::vector<float> velocities;
    std::vector<int64_t> ids;
    splitPV(state->domain, local(), *positions, velocities, ids);

    XDMF::VertexGrid grid(positions, comm);

    std::vector<XDMF::Channel> channels;
    channels.push_back(XDMF::Channel("velocity", velocities.data(),
                                     XDMF::Channel::DataForm::Vector, XDMF::Channel::NumberType::Float, DataTypeWrapper<float>() ));
    channels.push_back(XDMF::Channel(ChannelNames::globalIds, ids.data(),
                                     XDMF::Channel::DataForm::Scalar, XDMF::Channel::NumberType::Int64, DataTypeWrapper<int64_t>() ));

    // do not dump velocities, they are already there
    _extractPersistentExtraParticleData(channels, {ChannelNames::velocities});
    
    XDMF::write(filename, &grid, channels, comm);

    createCheckpointSymlink(comm, path, "PV", "xmf", checkpointId);
    
    debug("Checkpoint for particle vector '%s' successfully written", name.c_str());
}

void ParticleVector::_getRestartExchangeMap(MPI_Comm comm, const std::vector<float4> &pos, std::vector<int>& map)
{
    int dims[3], periods[3], coords[3];
    MPI_Check( MPI_Cart_get(comm, 3, dims, periods, coords) );

    map.resize(pos.size());
    
    for (int i = 0; i < pos.size(); ++i) {
        const auto& r = make_float3(pos[i]);
        int3 procId3 = make_int3(floorf(r / state->domain.localSize));

        if (procId3.x >= dims[0] || procId3.y >= dims[1] || procId3.z >= dims[2]) {
            map[i] = RestartHelpers::InvalidProc;
            continue;
        }
        
        int procId;
        MPI_Check( MPI_Cart_rank(comm, (int*)&procId3, &procId) );
        map[i] = procId;

        int rank;
        MPI_Comm_rank(comm, &rank);
    }
}

std::vector<int> ParticleVector::_restartParticleData(MPI_Comm comm, std::string path)
{
    CUDA_Check( cudaDeviceSynchronize() );
    
    auto filename = createCheckpointName(path, "PV", "xmf");
    info("Restarting particle vector %s from file %s", name.c_str(), filename.c_str());

    XDMF::readParticleData(filename, comm, this);

    std::vector<float4> pos4(local()->size()), vel4(local()->size());
    std::copy(local()->positions() .begin(), local()->positions() .end(), pos4.begin());
    std::copy(local()->velocities().begin(), local()->velocities().end(), vel4.begin());

    std::vector<int> map;
    
    _getRestartExchangeMap(comm, pos4, map);
    RestartHelpers::exchangeData(comm, map, pos4, 1);
    RestartHelpers::exchangeData(comm, map, vel4, 1);
    RestartHelpers::copyShiftCoordinates(state->domain, pos4, vel4, local());

    local()->positions ().uploadToDevice(defaultStream);
    local()->velocities().uploadToDevice(defaultStream);
    CUDA_Check( cudaDeviceSynchronize() );

    info("Successfully read %d particles", local()->size());

    return map;
}

void ParticleVector::checkpoint(MPI_Comm comm, std::string path, int checkpointId)
{
    _checkpointParticleData(comm, path, checkpointId);
}

void ParticleVector::restart(MPI_Comm comm, std::string path)
{
    _restartParticleData(comm, path);
}


