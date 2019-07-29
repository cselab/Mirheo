#include "particle_vector.h"
#include "checkpoint/helpers.h"
#include "restart/helpers.h"

#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/xdmf/type_map.h>
#include <core/xdmf/xdmf.h>

#include <mpi.h>

constexpr const char *RestartPVIdentifier = "PV";

LocalParticleVector::LocalParticleVector(ParticleVector *pv, int n) :
    pv(pv)
{
    dataPerParticle.createData<float4>(ChannelNames::positions,  n);
    dataPerParticle.createData<float4>(ChannelNames::velocities, n);
    dataPerParticle.createData<Force>(ChannelNames::forces, n);

    // positions are treated specially, do not need to be persistent
    dataPerParticle.setPersistenceMode(ChannelNames::velocities, DataManager::PersistenceMode::Active);
    dataPerParticle.setShiftMode(ChannelNames::positions, DataManager::ShiftMode::Active);
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

ParticleVector::ParticleVector(const MirState *state, std::string name, float mass, int n) :
    ParticleVector(state, name, mass,
                   std::make_unique<LocalParticleVector>(this, n),
                   std::make_unique<LocalParticleVector>(this, 0) )
{}

ParticleVector::ParticleVector(const MirState *state, std::string name,  float mass,
                               std::unique_ptr<LocalParticleVector>&& local,
                               std::unique_ptr<LocalParticleVector>&& halo) :
    MirSimulationObject(state, name),
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

void ParticleVector::_checkpointParticleData(MPI_Comm comm, std::string path, int checkpointId)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointNameWithId(path, RestartPVIdentifier, "", checkpointId);
    info("Checkpoint for particle vector '%s', writing to file %s",
         name.c_str(), filename.c_str());

    auto& pos4 = local()->positions ();
    auto& vel4 = local()->velocities();
    
    pos4.downloadFromDevice(defaultStream, ContainersSynch::Asynch);
    vel4.downloadFromDevice(defaultStream, ContainersSynch::Synch);

    auto positions = std::make_shared<std::vector<float3>>();
    std::vector<float3> velocities;
    std::vector<int64_t> ids;
    
    std::tie(*positions, velocities, ids) = CheckpointHelpers::splitAndShiftPosVel(state->domain,
                                                                                   pos4, vel4);

    XDMF::VertexGrid grid(positions, comm);

    // do not dump velocities, they are already there
    const std::set<std::string> blackList {{ChannelNames::velocities}};

    auto channels = CheckpointHelpers::extractShiftPersistentData(state->domain,
                                                                  local()->dataPerParticle,
                                                                  blackList);

    channels.emplace_back(ChannelNames::XDMF::velocity, velocities.data(),
                          XDMF::Channel::DataForm::Vector,
                          XDMF::Channel::NumberType::Float,
                          DataTypeWrapper<float>());
    
    channels.emplace_back(ChannelNames::XDMF::ids, ids.data(),
                          XDMF::Channel::DataForm::Scalar,
                          XDMF::Channel::NumberType::Int64,
                          DataTypeWrapper<int64_t>());

    XDMF::write(filename, &grid, channels, comm);

    createCheckpointSymlink(comm, path, RestartPVIdentifier, "xmf", checkpointId);
    
    debug("Checkpoint for particle vector '%s' successfully written", name.c_str());
}

ParticleVector::ExchMapSize ParticleVector::_restartParticleData(MPI_Comm comm, std::string path,
                                                                 int chunkSize)
{
    CUDA_Check( cudaDeviceSynchronize() );
    
    auto filename = createCheckpointName(path, RestartPVIdentifier, "xmf");
    info("Restarting particle data from file %s", name.c_str(), filename.c_str());

    auto listData = RestartHelpers::readData(filename, comm, chunkSize);

    auto pos = RestartHelpers::extractChannel<float3> (ChannelNames::XDMF::position, listData);
    auto vel = RestartHelpers::extractChannel<float3> (ChannelNames::XDMF::velocity, listData);
    auto ids = RestartHelpers::extractChannel<int64_t>(ChannelNames::XDMF::ids,      listData);
    
    std::vector<float4> pos4, vel4;
    std::tie(pos4, vel4) = RestartHelpers::combinePosVelIds(pos, vel, ids);

    auto map = RestartHelpers::getExchangeMap(comm, state->domain, chunkSize, pos);

    RestartHelpers::exchangeData(comm, map, pos4, chunkSize);
    RestartHelpers::exchangeData(comm, map, vel4, chunkSize);
    RestartHelpers::exchangeListData(comm, map, listData, chunkSize);

    const int newSize = pos4.size() / chunkSize;

    auto& dataPerParticle = local()->dataPerParticle;
    dataPerParticle.resize_anew(newSize * chunkSize);

    auto& positions  = local()->positions();
    auto& velocities = local()->velocities();

    RestartHelpers::shiftElementsGlobal2Local(pos4, state->domain);
    
    std::copy(pos4.begin(), pos4.end(), positions .begin());
    std::copy(vel4.begin(), vel4.end(), velocities.begin());

    positions .uploadToDevice(defaultStream);
    velocities.uploadToDevice(defaultStream);

    for (auto& entry : listData)
    {
        auto channelDesc = &dataPerParticle.getChannelDescOrDie(entry.name);
        
        mpark::visit([&](const auto& data)
        {
            using T = typename std::remove_reference<decltype(data)>::type::value_type;
            auto dstPtr = dataPerParticle.getData<T>(entry.name);

            if (channelDesc->needShift())
                RestartHelpers::shiftElementsGlobal2Local(data, state->domain);

            std::copy(data.begin(), data.end(), dstPtr->begin());
            dstPtr->uploadToDevice(defaultStream);
            
        }, entry.data);
    }
    
    return {map, newSize};
}

void ParticleVector::checkpoint(MPI_Comm comm, std::string path, int checkpointId)
{
    _checkpointParticleData(comm, path, checkpointId);
}

void ParticleVector::restart(MPI_Comm comm, std::string path)
{
    constexpr int particleChunkSize = 1;
    auto ms = _restartParticleData(comm, path, particleChunkSize);
    local()->resize(ms.newSize, defaultStream);
}


