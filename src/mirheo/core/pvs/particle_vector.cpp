#include "particle_vector.h"
#include "checkpoint/helpers.h"
#include "restart/helpers.h"

#include <mirheo/core/snapshot.h>
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
    pv_(pv)
{
    dataPerParticle.createData<real4>(channel_names::positions,  numParts);
    dataPerParticle.createData<real4>(channel_names::velocities, numParts);
    dataPerParticle.createData<Force>(channel_names::forces, numParts);

    dataPerParticle.setPersistenceMode(channel_names::positions,  DataManager::PersistenceMode::Active);
    dataPerParticle.setShiftMode      (channel_names::positions,  DataManager::ShiftMode::Active);
    dataPerParticle.setPersistenceMode(channel_names::velocities, DataManager::PersistenceMode::Active);
    resize_anew(numParts);
}

LocalParticleVector::~LocalParticleVector() = default;

void swap(LocalParticleVector& a, LocalParticleVector &b)
{
    std::swap(a.pv_, b.pv_);
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
    return * dataPerParticle.getData<real4>(channel_names::positions);
}

PinnedBuffer<real4>& LocalParticleVector::velocities()
{
    return * dataPerParticle.getData<real4>(channel_names::velocities);
}

PinnedBuffer<Force>& LocalParticleVector::forces()
{
    return * dataPerParticle.getData<Force>(channel_names::forces);
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
    mass_(mass),
    local_(std::move(local)),
    halo_(std::move(halo))
{
    // old positions and velocities don't need to exchanged in general
    requireDataPerParticle<real4> (channel_names::oldPositions, DataManager::PersistenceMode::None);
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

py_types::VectorOfReal3 ParticleVector::getCoordinates_vector()
{
    auto& pos = local()->positions();
    pos.downloadFromDevice(defaultStream);

    py_types::VectorOfReal3 res(pos.size());
    for (size_t i = 0; i < pos.size(); i++)
    {
        real3 r = make_real3(pos[i]);
        r = getState()->domain.local2global(r);
        res[i] = { r.x, r.y, r.z };
    }

    return res;
}

py_types::VectorOfReal3 ParticleVector::getVelocities_vector()
{
    auto& vel = local()->velocities();
    vel.downloadFromDevice(defaultStream);

    py_types::VectorOfReal3 res(vel.size());
    for (size_t i = 0; i < vel.size(); i++)
    {
        real3 u = make_real3(vel[i]);
        res[i] = { u.x, u.y, u.z };
    }

    return res;
}

py_types::VectorOfReal3 ParticleVector::getForces_vector()
{
    HostBuffer<Force> forces;
    forces.copy(local()->forces(), defaultStream);

    py_types::VectorOfReal3 res(forces.size());
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

void ParticleVector::_snapshotParticleData(MPI_Comm comm, const std::string& filename)
{
    CUDA_Check( cudaDeviceSynchronize() );

    info("Checkpoint for particle vector '%s', writing to file %s",
         getCName(), filename.c_str());

    auto& pos4 = local()->positions ();
    auto& vel4 = local()->velocities();

    pos4.downloadFromDevice(defaultStream, ContainersSynch::Asynch);
    vel4.downloadFromDevice(defaultStream, ContainersSynch::Synch);

    auto positions = std::make_shared<std::vector<real3>>();
    std::vector<real3> velocities;
    std::vector<int64_t> ids;

    std::tie(*positions, velocities, ids) = checkpoint_helpers::splitAndShiftPosVel(getState()->domain,
                                                                                   pos4, vel4);

    XDMF::VertexGrid grid(positions, comm);

    // do not dump positions and velocities, they are already there
    const std::set<std::string> blackList {{channel_names::positions, channel_names::velocities}};

    auto channels = checkpoint_helpers::extractShiftPersistentData(getState()->domain,
                                                                  local()->dataPerParticle,
                                                                  blackList);

    channels.push_back(XDMF::Channel{channel_names::XDMF::velocity, velocities.data(),
                                         XDMF::Channel::DataForm::Vector,
                                         XDMF::getNumberType<real>(),
                                         DataTypeWrapper<real>(),
                                         XDMF::Channel::NeedShift::False});

    channels.push_back(XDMF::Channel{channel_names::XDMF::ids, ids.data(),
                                         XDMF::Channel::DataForm::Scalar,
                                         XDMF::Channel::NumberType::Int64,
                                         DataTypeWrapper<int64_t>(),
                                         XDMF::Channel::NeedShift::False});

    XDMF::write(filename, &grid, channels, comm);

    debug("Checkpoint for particle vector '%s' successfully written", getCName());
}

void ParticleVector::_checkpointParticleData(MPI_Comm comm, const std::string& path, int checkpointId)
{
    auto filename = createCheckpointNameWithId(path, RestartPVIdentifier, "", checkpointId);
    _snapshotParticleData(comm, filename);

    createCheckpointSymlink(comm, path, RestartPVIdentifier, "xmf", checkpointId);
    debug("Created a symlink for particle vector '%s'", getCName());
}

ParticleVector::ExchMapSize ParticleVector::_restartParticleData(MPI_Comm comm, const std::string& path,
                                                                 int chunkSize)
{
    CUDA_Check( cudaDeviceSynchronize() );

    const auto filename = createCheckpointName(path, RestartPVIdentifier, "xmf");
    info("Restarting particle vector %s data from file %s", getCName(), filename.c_str());

    auto listData = restart_helpers::readData(filename, comm, chunkSize);

    auto pos = restart_helpers::extractChannel<real3>  (channel_names::XDMF::position, listData);
    auto vel = restart_helpers::extractChannel<real3>  (channel_names::XDMF::velocity, listData);
    auto ids = restart_helpers::extractChannel<int64_t>(channel_names::XDMF::ids,      listData);

    std::vector<real4> pos4, vel4;
    std::tie(pos4, vel4) = restart_helpers::combinePosVelIds(pos, vel, ids);

    const auto map = restart_helpers::getExchangeMap(comm, getState()->domain, chunkSize, pos);

    restart_helpers::exchangeData(comm, map, pos4, chunkSize);
    restart_helpers::exchangeData(comm, map, vel4, chunkSize);
    restart_helpers::exchangeListData(comm, map, listData, chunkSize);
    restart_helpers::requireExtraDataPerParticle(listData, this);

    const int newSize = static_cast<int>(pos4.size()) / chunkSize;

    auto& dataPerParticle = local()->dataPerParticle;
    dataPerParticle.resize_anew(newSize * chunkSize);

    auto& positions  = local()->positions();
    auto& velocities = local()->velocities();

    restart_helpers::shiftElementsGlobal2Local(pos4, getState()->domain);

    std::copy(pos4.begin(), pos4.end(), positions .begin());
    std::copy(vel4.begin(), vel4.end(), velocities.begin());

    positions .uploadToDevice(defaultStream);
    velocities.uploadToDevice(defaultStream);

    restart_helpers::copyAndShiftListData(getState()->domain, listData, dataPerParticle);

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

void ParticleVector::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<ParticleVector>(this, _saveSnapshot(saver, "ParticleVector"));
}

ConfigObject ParticleVector::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    // The filename does not include the extension.
    std::string filename = joinPaths(saver.getContext().path, getName() + "." + RestartPVIdentifier);
    _snapshotParticleData(saver.getContext().groupComm, filename);
    ConfigObject config = MirSimulationObject::_saveSnapshot(
            saver, "ParticleVector", typeName);
    config.emplace("mass", saver(mass_));
    return config;
}

ParticleVector::ParticleVector(const MirState *state, Loader&, const ConfigObject& config) :
    ParticleVector{state, (const std::string&)config["name"], (real)config["mass"]}
{
    assert(config["__type"].getString() == "ParticleVector");
    // Note: Particles loaded by RestartIC.
}

} // namespace mirheo
