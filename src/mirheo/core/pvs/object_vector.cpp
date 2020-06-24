#include "object_vector.h"

#include "checkpoint/helpers.h"
#include "restart/helpers.h"
#include "utils/compute_com_extents.h"

#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/xdmf/xdmf.h>

#include <limits>

namespace mirheo
{

constexpr const char *RestartOVIdentifier = "OV";

LocalObjectVector::LocalObjectVector(ParticleVector *pv, int objSize, int nObjects) :
    LocalParticleVector(pv, objSize*nObjects),
    objSize_(objSize),
    nObjects_(nObjects)
{
    if (objSize_ <= 0)
        die("Object vector should contain at least one particle per object instead of %d", objSize_);

    resize_anew(nObjects_ * objSize_);
}

LocalObjectVector::~LocalObjectVector() = default;

void swap(LocalObjectVector& a, LocalObjectVector& b)
{
    swap(static_cast<LocalParticleVector &>(a), static_cast<LocalParticleVector &>(b));
    std::swap(a.nObjects_, b.nObjects_);
    std::swap(a.objSize_,  b.objSize_);
    swap(a.dataPerObject, b.dataPerObject);
}

void LocalObjectVector::resize(int np, cudaStream_t stream)
{
    nObjects_ = _computeNobjects(np);
    LocalParticleVector::resize(np, stream);
    dataPerObject.resize(nObjects_, stream);
}

void LocalObjectVector::resize_anew(int np)
{
    nObjects_ = _computeNobjects(np);
    LocalParticleVector::resize_anew(np);
    dataPerObject.resize_anew(nObjects_);
}

void LocalObjectVector::computeGlobalIds(MPI_Comm comm, cudaStream_t stream)
{
    LocalParticleVector::computeGlobalIds(comm, stream);

    if (size() == 0) return;

    Particle p0( positions()[0], velocities()[0]);
    int64_t rankStart = p0.getId();

    if ((rankStart % objSize_) != 0)
        die("Something went wrong when computing ids of '%s':"
            "got rankStart = '%ld' while objectSize is '%d'",
            parent()->getCName(), rankStart, objSize_);

    auto& ids = *dataPerObject.getData<int64_t>(channel_names::globalIds);
    int64_t id = (int64_t) (rankStart / objSize_);

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
    return dataPerParticle.getData<real4>(channel_names::oldPositions);
}

PinnedBuffer<Force>* LocalObjectVector::getMeshForces(__UNUSED cudaStream_t stream)
{
    return &forces();
}

int LocalObjectVector::getObjectSize() const
{
    return objSize_;
}

int LocalObjectVector::getNumObjects() const
{
    return nObjects_;
}

int LocalObjectVector::_computeNobjects(int np) const
{
    if (np % objSize_ != 0)
        die("Incorrect number of particles in object: given %d, must be a multiple of %d", np, objSize_);

    return np / objSize_;
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
    objSize_(objSize)
{
    // center of mass and extents are not to be sent around
    // it's cheaper to compute them on site
    requireDataPerObject<COMandExtent>(channel_names::comExtents, DataManager::PersistenceMode::None);

    // object ids must always follow objects
    requireDataPerObject<int64_t>(channel_names::globalIds, DataManager::PersistenceMode::Active);
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

    auto coms_extents = local()->dataPerObject.getData<COMandExtent>(channel_names::comExtents);

    coms_extents->downloadFromDevice(defaultStream, ContainersSynch::Synch);

    auto positions = std::make_shared<std::vector<real3>>(getCom(getState()->domain, *coms_extents));

    XDMF::VertexGrid grid(positions, comm);

    auto channels = checkpoint_helpers::extractShiftPersistentData(getState()->domain,
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

    auto listData = restart_helpers::readData(filename, comm, objChunkSize);

    // remove positions from the read data (artificial for non rov)
    restart_helpers::extractChannel<real3> (channel_names::XDMF::position, listData);

    restart_helpers::exchangeListData(comm, ms.map, listData, objChunkSize);
    restart_helpers::requireExtraDataPerObject(listData, this);

    auto& dataPerObject = local()->dataPerObject;
    dataPerObject.resize_anew(ms.newSize);

    restart_helpers::copyAndShiftListData(getState()->domain, listData, dataPerObject);

    info("Successfully read object infos of '%s'", getCName());
}

void ObjectVector::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    _checkpointParticleData(comm, path, checkpointId);
    _checkpointObjectData  (comm, path, checkpointId);
}

void ObjectVector::restart(MPI_Comm comm, const std::string& path)
{
    const auto ms = _restartParticleData(comm, path, getObjectSize());
    _restartObjectData(comm, path, ms);

    local()->resize(ms.newSize * getObjectSize(), defaultStream);
}

int ObjectVector::getObjectSize() const
{
    return objSize_;
}

void ObjectVector::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<ObjectVector>(this, _saveSnapshot(saver, "ObjectVector"));
}

ConfigObject ObjectVector::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    // The filename does not include the extension.
    std::string filename = joinPaths(saver.getContext().path, getName() + "." + RestartOVIdentifier);
    _snapshotObjectData(saver.getContext().groupComm, filename);

    ConfigObject config = ParticleVector::_saveSnapshot(saver, typeName);
    config.emplace("objSize", saver(objSize_));
    config.emplace("mesh",    saver(mesh));
    return config;
}

} // namespace mirheo
