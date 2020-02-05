#include "particle_vector.h"
#include "checkpoint/helpers.h"
#include "restart/helpers.h"

#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/xdmf/type_map.h>
#include <mirheo/core/xdmf/xdmf.h>

#include <mpi.h>

namespace mirheo
{

constexpr const char *RestartPVIdentifier = "PV";

std::string getParticleVectorLocalityStr(ParticleVectorLocality locality)
{
    if (locality == ParticleVectorLocality::Local)
        return "local";
    else
        return "halo";
}

LocalParticleVector::LocalParticleVector(ParticleVector *pv, int numParts) :
    pv(pv)
{
    dataPerParticle.createData<real4>(ChannelNames::positions,  numParts);
    dataPerParticle.createData<real4>(ChannelNames::velocities, numParts);
    dataPerParticle.createData<Force>(ChannelNames::forces, numParts);

    dataPerParticle.setPersistenceMode(ChannelNames::positions,  DataManager::PersistenceMode::Active);
    dataPerParticle.setShiftMode      (ChannelNames::positions,  DataManager::ShiftMode::Active);
    dataPerParticle.setPersistenceMode(ChannelNames::velocities, DataManager::PersistenceMode::Active);
    resize_anew(numParts);
}

LocalParticleVector::~LocalParticleVector() = default;

void swap(LocalParticleVector& a, LocalParticleVector &b)
{
    std::swap(a.pv, b.pv);
    swap(a.dataPerParticle, b.dataPerParticle);
    std::swap(a.np_, b.np_);
}

void LocalParticleVector::resize(int np, cudaStream_t stream)
{
    if (np < 0) die("Tried to resize PV to %d < 0 particles", np);
    dataPerParticle.resize(np, stream);
    np_ = np;
}

void LocalParticleVector::resize_anew(int np)
{
    if (np < 0) die("Tried to resize PV to %d < 0 particles", np);
    dataPerParticle.resize_anew(np);
    np_ = np;
}

PinnedBuffer<real4>& LocalParticleVector::positions()
{
    return * dataPerParticle.getData<real4>(ChannelNames::positions);
}

PinnedBuffer<real4>& LocalParticleVector::velocities()
{
    return * dataPerParticle.getData<real4>(ChannelNames::velocities);
}

PinnedBuffer<Force>& LocalParticleVector::forces()
{
    return * dataPerParticle.getData<Force>(ChannelNames::forces);
}

void LocalParticleVector::computeGlobalIds(MPI_Comm comm, cudaStream_t stream)
{
    int64_t rankStart = 0;
    int64_t np64 = np_;
        
    MPI_Check( MPI_Exscan(&np64, &rankStart, 1, MPI_INT64_T, MPI_SUM, comm) );

    auto& pos = positions();
    auto& vel = velocities();
    
    pos.downloadFromDevice(stream, ContainersSynch::Asynch);
    vel.downloadFromDevice(stream);

    int64_t id = rankStart;
    for (size_t i = 0; i < pos.size(); ++i)
    {
        Particle p(pos[i], vel[i]);
        p.setId(id++);
        pos[i] = p.r2Real4();
        vel[i] = p.u2Real4();
    }
    
    pos.uploadToDevice(stream);
    vel.uploadToDevice(stream);
}


//============================================================================
// Particle Vector
//============================================================================

ParticleVector::ParticleVector(const MirState *state, const std::string& name, real mass, int n) :
    ParticleVector(state, name, mass,
                   std::make_unique<LocalParticleVector>(this, n),
                   std::make_unique<LocalParticleVector>(this, 0) )
{}

ParticleVector::ParticleVector(const MirState *state, const std::string& name, real mass,
                               std::unique_ptr<LocalParticleVector>&& local,
                               std::unique_ptr<LocalParticleVector>&& halo) :
    MirSimulationObject(state, name),
    mass(mass),
    local_(std::move(local)),
    halo_(std::move(halo))
{
    // old positions and velocities don't need to exchanged in general
    requireDataPerParticle<real4> (ChannelNames::oldPositions, DataManager::PersistenceMode::None);
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

PyTypes::VectorOfReal3 ParticleVector::getCoordinates_vector()
{
    auto& pos = local()->positions();
    pos.downloadFromDevice(defaultStream);
    
    PyTypes::VectorOfReal3 res(pos.size());
    for (size_t i = 0; i < pos.size(); i++)
    {
        real3 r = make_real3(pos[i]);
        r = getState()->domain.local2global(r);
        res[i] = { r.x, r.y, r.z };
    }
    
    return res;
}

PyTypes::VectorOfReal3 ParticleVector::getVelocities_vector()
{
    auto& vel = local()->velocities();
    vel.downloadFromDevice(defaultStream);
    
    PyTypes::VectorOfReal3 res(vel.size());
    for (size_t i = 0; i < vel.size(); i++)
    {
        real3 u = make_real3(vel[i]);
        res[i] = { u.x, u.y, u.z };
    }
    
    return res;
}

PyTypes::VectorOfReal3 ParticleVector::getForces_vector()
{
    HostBuffer<Force> forces;
    forces.copy(local()->forces(), defaultStream);
    
    PyTypes::VectorOfReal3 res(forces.size());
    for (size_t i = 0; i < forces.size(); i++)
    {
        real3 f = forces[i].f;
        res[i] = { f.x, f.y, f.z };
    }
    
    return res;
}

void ParticleVector::setCoordinates_vector(const std::vector<real3>& coordinates)
{
    auto& pos = local()->positions();
    const size_t n = local()->size();
    
    if (coordinates.size() != n)
        throw std::invalid_argument("Wrong number of particles passed, "
            "expected: " + std::to_string(n) +
            ", got: " + std::to_string(coordinates.size()) );
    
    for (size_t i = 0; i < coordinates.size(); i++)
    {
        real3 r = coordinates[i];
        r = getState()->domain.global2local(r);
        pos[i].x = r.x;
        pos[i].y = r.y;
        pos[i].z = r.z;
    }
    
    pos.uploadToDevice(defaultStream);
}

void ParticleVector::setVelocities_vector(const std::vector<real3>& velocities)
{
    auto& vel = local()->velocities();
    const size_t n = local()->size();
    
    if (velocities.size() != n)
        throw std::invalid_argument("Wrong number of particles passed, "
        "expected: " + std::to_string(n) +
        ", got: " + std::to_string(velocities.size()) );
    
    for (size_t i = 0; i < velocities.size(); i++)
    {
        const real3 u = velocities[i];
        vel[i].x = u.x;
        vel[i].y = u.y;
        vel[i].z = u.z;
    }
    
    vel.uploadToDevice(defaultStream);
}

void ParticleVector::setForces_vector(const std::vector<real3>& forces)
{
    HostBuffer<Force> myforces(local()->size());
    const size_t n = local()->size();
    if (forces.size() != n)
        throw std::invalid_argument("Wrong number of particles passed, "
        "expected: " + std::to_string(n) +
        ", got: " + std::to_string(forces.size()) );
    
    for (size_t i = 0; i < forces.size(); i++)
        myforces[i].f = forces[i];
    
    local()->forces().copy(myforces);
    local()->forces().uploadToDevice(defaultStream);
}

void ParticleVector::_checkpointParticleData(MPI_Comm comm, const std::string& path, int checkpointId)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointNameWithId(path, RestartPVIdentifier, "", checkpointId);
    info("Checkpoint for particle vector '%s', writing to file %s",
         getCName(), filename.c_str());

    auto& pos4 = local()->positions ();
    auto& vel4 = local()->velocities();
    
    pos4.downloadFromDevice(defaultStream, ContainersSynch::Asynch);
    vel4.downloadFromDevice(defaultStream, ContainersSynch::Synch);

    auto positions = std::make_shared<std::vector<real3>>();
    std::vector<real3> velocities;
    std::vector<int64_t> ids;
    
    std::tie(*positions, velocities, ids) = CheckpointHelpers::splitAndShiftPosVel(getState()->domain,
                                                                                   pos4, vel4);

    XDMF::VertexGrid grid(positions, comm);

    // do not dump positions and velocities, they are already there
    const std::set<std::string> blackList {{ChannelNames::positions, ChannelNames::velocities}};

    auto channels = CheckpointHelpers::extractShiftPersistentData(getState()->domain,
                                                                  local()->dataPerParticle,
                                                                  blackList);

    channels.push_back(XDMF::Channel{ChannelNames::XDMF::velocity, velocities.data(),
                                         XDMF::Channel::DataForm::Vector,
                                         XDMF::getNumberType<real>(),
                                         DataTypeWrapper<real>(),
                                         XDMF::Channel::NeedShift::False});
    
    channels.push_back(XDMF::Channel{ChannelNames::XDMF::ids, ids.data(),
                                         XDMF::Channel::DataForm::Scalar,
                                         XDMF::Channel::NumberType::Int64,
                                         DataTypeWrapper<int64_t>(),
                                         XDMF::Channel::NeedShift::False});

    XDMF::write(filename, &grid, channels, comm);

    createCheckpointSymlink(comm, path, RestartPVIdentifier, "xmf", checkpointId);
    
    debug("Checkpoint for particle vector '%s' successfully written", getCName());
}

ParticleVector::ExchMapSize ParticleVector::_restartParticleData(MPI_Comm comm, const std::string& path,
                                                                 int chunkSize)
{
    CUDA_Check( cudaDeviceSynchronize() );
    
    const auto filename = createCheckpointName(path, RestartPVIdentifier, "xmf");
    info("Restarting particle data from file %s", getCName(), filename.c_str());

    auto listData = RestartHelpers::readData(filename, comm, chunkSize);

    auto pos = RestartHelpers::extractChannel<real3>  (ChannelNames::XDMF::position, listData);
    auto vel = RestartHelpers::extractChannel<real3>  (ChannelNames::XDMF::velocity, listData);
    auto ids = RestartHelpers::extractChannel<int64_t>(ChannelNames::XDMF::ids,      listData);
    
    std::vector<real4> pos4, vel4;
    std::tie(pos4, vel4) = RestartHelpers::combinePosVelIds(pos, vel, ids);

    const auto map = RestartHelpers::getExchangeMap(comm, getState()->domain, chunkSize, pos);

    RestartHelpers::exchangeData(comm, map, pos4, chunkSize);
    RestartHelpers::exchangeData(comm, map, vel4, chunkSize);
    RestartHelpers::exchangeListData(comm, map, listData, chunkSize);
    RestartHelpers::requireExtraDataPerParticle(listData, this);
    
    const int newSize = static_cast<int>(pos4.size()) / chunkSize;

    auto& dataPerParticle = local()->dataPerParticle;
    dataPerParticle.resize_anew(newSize * chunkSize);

    auto& positions  = local()->positions();
    auto& velocities = local()->velocities();

    RestartHelpers::shiftElementsGlobal2Local(pos4, getState()->domain);
    
    std::copy(pos4.begin(), pos4.end(), positions .begin());
    std::copy(vel4.begin(), vel4.end(), velocities.begin());

    positions .uploadToDevice(defaultStream);
    velocities.uploadToDevice(defaultStream);

    RestartHelpers::copyAndShiftListData(getState()->domain, listData, dataPerParticle);

    return {map, newSize};
}

void ParticleVector::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    _checkpointParticleData(comm, path, checkpointId);
}

void ParticleVector::restart(MPI_Comm comm, const std::string& path)
{
    constexpr int particleChunkSize = 1;
    const auto ms = _restartParticleData(comm, path, particleChunkSize);
    local()->resize(ms.newSize, defaultStream);
}

ConfigDictionary ParticleVector::writeSnapshot(Dumper& dumper) const
{
    return {
        {"__category", dumper("ParticleVector")},
        {"__type",     dumper("ParticleVector")},
        {"mass",       dumper(mass)},
    };
}

} // namespace mirheo
