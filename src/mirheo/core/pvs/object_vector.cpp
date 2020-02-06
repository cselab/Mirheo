#include "object_vector.h"

#include "checkpoint/helpers.h"
#include "restart/helpers.h"
#include "utils/compute_com_extents.h"

#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/xdmf/xdmf.h>

#include <limits>

namespace mirheo
{

constexpr const char *RestartOVIdentifier = "OV";

LocalObjectVector::LocalObjectVector(ParticleVector *pv, int objSize, int nObjects) :
    LocalParticleVector(pv, objSize*nObjects),
    nObjects(nObjects),
    objSize(objSize)
{
    if (objSize <= 0)
        die("Object vector should contain at least one particle per object instead of %d", objSize);

    resize_anew(nObjects*objSize);
}

LocalObjectVector::~LocalObjectVector() = default;

void swap(LocalObjectVector& a, LocalObjectVector& b)
{
    swap(static_cast<LocalParticleVector &>(a), static_cast<LocalParticleVector &>(b));
    std::swap(a.nObjects, b.nObjects);
    std::swap(a.objSize,  b.objSize);
    swap(a.dataPerObject, b.dataPerObject);
}

void LocalObjectVector::resize(int np, cudaStream_t stream)
{
    nObjects = getNobjects(np);
    LocalParticleVector::resize(np, stream);
    dataPerObject.resize(nObjects, stream);
}

void LocalObjectVector::resize_anew(int np)
{
    nObjects = getNobjects(np);
    LocalParticleVector::resize_anew(np);
    dataPerObject.resize_anew(nObjects);
}

void LocalObjectVector::computeGlobalIds(MPI_Comm comm, cudaStream_t stream)
{
    LocalParticleVector::computeGlobalIds(comm, stream);

    if (size() == 0) return;

    Particle p0( positions()[0], velocities()[0]);
    int64_t rankStart = p0.getId();
    
    if ((rankStart % objSize) != 0)
        die("Something went wrong when computing ids of '%s':"
            "got rankStart = '%ld' while objectSize is '%d'",
            pv->getCName(), rankStart, objSize);

    auto& ids = *dataPerObject.getData<int64_t>(ChannelNames::globalIds);
    int64_t id = (int64_t) (rankStart / objSize);
    
    for (auto& i : ids)
        i = id++;

    ids.uploadToDevice(stream);
}

PinnedBuffer<real4>* LocalObjectVector::getMeshVertices(__UNUSED cudaStream_t stream)
{
    return &positions();
}

PinnedBuffer<real4>* LocalObjectVector::getOldMeshVertices(__UNUSED cudaStream_t stream)
{
    return dataPerParticle.getData<real4>(ChannelNames::oldPositions);
}

PinnedBuffer<Force>* LocalObjectVector::getMeshForces(__UNUSED cudaStream_t stream)
{
    return &forces();
}

int LocalObjectVector::getNobjects(int np) const
{
    if (np % objSize != 0)
        die("Incorrect number of particles in object: given %d, must be a multiple of %d", np, objSize);

    return np / objSize;
}


ObjectVector::ObjectVector(const MirState *state, const std::string& name, real mass, int objSize, int nObjects) :
    ObjectVector( state, name, mass, objSize,
                  std::make_unique<LocalObjectVector>(this, objSize, nObjects),
                  std::make_unique<LocalObjectVector>(this, objSize, 0) )
{}

ObjectVector::ObjectVector(const MirState *state, const std::string& name, real mass, int objSize,
                           std::unique_ptr<LocalParticleVector>&& local,
                           std::unique_ptr<LocalParticleVector>&& halo) :
    ParticleVector(state, name, mass, std::move(local), std::move(halo)),
    objSize(objSize)
{
    // center of mass and extents are not to be sent around
    // it's cheaper to compute them on site
    requireDataPerObject<COMandExtent>(ChannelNames::comExtents, DataManager::PersistenceMode::None);

    // object ids must always follow objects
    requireDataPerObject<int64_t>(ChannelNames::globalIds, DataManager::PersistenceMode::Active);
}

ObjectVector::~ObjectVector() = default;

void ObjectVector::findExtentAndCOM(cudaStream_t stream, ParticleVectorLocality locality)
{
    auto lov = get(locality);

    debug("Computing COM and extent OV '%s' (%s)",
          getCName(), getParticleVectorLocalityStr(locality).c_str());

    computeComExtents(this, lov, stream);
}

static std::vector<real3> getCom(DomainInfo domain,
                                  const PinnedBuffer<COMandExtent>& com_extents)
{
    const size_t n = com_extents.size();
    std::vector<real3> pos(n);

    for (size_t i = 0; i < n; ++i)
    {
        auto r = com_extents[i].com;
        pos[i] = domain.local2global(r);
    }

    return pos;
}

void ObjectVector::_snapshotObjectData(MPI_Comm comm, const std::string& filename)
{
    CUDA_Check( cudaDeviceSynchronize() );

    info("Checkpoint for object vector '%s', writing to file %s",
         getCName(), filename.c_str());

    auto coms_extents = local()->dataPerObject.getData<COMandExtent>(ChannelNames::comExtents);

    coms_extents->downloadFromDevice(defaultStream, ContainersSynch::Synch);
    
    auto positions = std::make_shared<std::vector<real3>>(getCom(getState()->domain, *coms_extents));

    XDMF::VertexGrid grid(positions, comm);

    auto channels = CheckpointHelpers::extractShiftPersistentData(getState()->domain,
                                                                  local()->dataPerObject);
    
    XDMF::write(filename, &grid, channels, comm);

    debug("Checkpoint for object vector '%s' successfully written", getCName());
}

void ObjectVector::_checkpointObjectData(MPI_Comm comm, const std::string& path, int checkpointId)
{
    auto filename = createCheckpointNameWithId(path, RestartOVIdentifier, "", checkpointId);
    _snapshotObjectData(comm, filename);
    createCheckpointSymlink(comm, path, RestartOVIdentifier, "xmf", checkpointId);
    debug("Created a symlink for object vector '%s'", getCName());
}

void ObjectVector::_restartObjectData(MPI_Comm comm, const std::string& path,
                                      const ObjectVector::ExchMapSize& ms)
{
    constexpr int objChunkSize = 1; // only one datum per object
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointName(path, RestartOVIdentifier, "xmf");
    info("Restarting object vector %s from file %s", getCName(), filename.c_str());

    auto listData = RestartHelpers::readData(filename, comm, objChunkSize);

    // remove positions from the read data (artificial for non rov)
    RestartHelpers::extractChannel<real3> (ChannelNames::XDMF::position, listData);
    
    RestartHelpers::exchangeListData(comm, ms.map, listData, objChunkSize);
    RestartHelpers::requireExtraDataPerObject(listData, this);

    auto& dataPerObject = local()->dataPerObject;
    dataPerObject.resize_anew(ms.newSize);

    RestartHelpers::copyAndShiftListData(getState()->domain, listData, dataPerObject);
    
    info("Successfully read object infos of '%s'", getCName());
}

void ObjectVector::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    _checkpointParticleData(comm, path, checkpointId);
    _checkpointObjectData  (comm, path, checkpointId);
}

void ObjectVector::restart(MPI_Comm comm, const std::string& path)
{
    const auto ms = _restartParticleData(comm, path, objSize);
    _restartObjectData(comm, path, ms);
    
    local()->resize(ms.newSize * objSize, defaultStream);
}

ConfigDictionary ObjectVector::writeSnapshot(Dumper &dumper)
{
    // The filename does not include the extension.
    std::string filename = joinPaths(dumper.getContext().path, getName() + "." + RestartOVIdentifier);
    _snapshotObjectData(dumper.getContext().groupComm, filename);

    ConfigDictionary dict = ParticleVector::writeSnapshot(dumper);
    dict.insert_or_assign("__type", dumper("ObjectVector"));
    dict.emplace("objSize",         dumper(objSize));
    dict.emplace("mesh",            dumper(mesh));
    return dict;
}

} // namespace mirheo
