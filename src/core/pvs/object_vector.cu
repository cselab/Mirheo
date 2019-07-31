#include "object_vector.h"
#include "views/ov.h"
#include "restart/helpers.h"
#include "checkpoint/helpers.h"

#include <core/utils/kernel_launch.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/xdmf/xdmf.h>

constexpr const char *RestartOVIdentifier = "OV";

namespace ObjectVectorKernels
{

__global__ void minMaxCom(OVview ovView)
{
    const int gid    = threadIdx.x + blockDim.x * blockIdx.x;
    const int objId  = gid / warpSize;
    const int laneId = gid % warpSize;
    if (objId >= ovView.nObjects) return;

    float3 mymin = make_float3( 1e+10f);
    float3 mymax = make_float3(-1e+10f);
    float3 mycom = make_float3(0);

#pragma unroll 3
    for (int i = laneId; i < ovView.objSize; i += warpSize)
    {
        const int offset = objId * ovView.objSize + i;

        const float3 coo = make_float3(ovView.readPosition(offset));

        mymin = fminf(mymin, coo);
        mymax = fmaxf(mymax, coo);
        mycom += coo;
    }

    mycom = warpReduce( mycom, [] (float a, float b) { return a+b; } );
    mymin = warpReduce( mymin, [] (float a, float b) { return fmin(a, b); } );
    mymax = warpReduce( mymax, [] (float a, float b) { return fmax(a, b); } );

    if (laneId == 0)
        ovView.comAndExtents[objId] = {mycom / ovView.objSize, mymin, mymax};
}

} // namespace ObjectVectorKernels


LocalObjectVector::LocalObjectVector(ParticleVector *pv, int objSize, int nObjects) :
    LocalParticleVector(pv, objSize*nObjects), objSize(objSize), nObjects(nObjects)
{
    if (objSize <= 0)
        die("Object vector should contain at least one particle per object instead of %d", objSize);

    resize_anew(nObjects*objSize);
}

LocalObjectVector::~LocalObjectVector() = default;

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

    if (np == 0) return;

    Particle p0( positions()[0], velocities()[0]);
    int64_t rankStart = p0.getId();
    
    if ((rankStart % objSize) != 0)
        die("Something went wrong when computing ids of '%s':"
            "got rankStart = '%ld' while objectSize is '%d'",
            pv->name.c_str(), rankStart, objSize);

    auto& ids = *dataPerObject.getData<int64_t>(ChannelNames::globalIds);
    int64_t id = (int64_t) (rankStart / objSize);
    
    for (auto& i : ids)
        i = id++;

    ids.uploadToDevice(stream);
}

PinnedBuffer<float4>* LocalObjectVector::getMeshVertices(cudaStream_t stream)
{
    return &positions();
}

PinnedBuffer<float4>* LocalObjectVector::getOldMeshVertices(cudaStream_t stream)
{
    return dataPerParticle.getData<float4>(ChannelNames::oldPositions);
}

PinnedBuffer<Force>* LocalObjectVector::getMeshForces(cudaStream_t stream)
{
    return &forces();
}

int LocalObjectVector::getNobjects(int np) const
{
    if (np % objSize != 0)
        die("Incorrect number of particles in object: given %d, must be a multiple of %d", np, objSize);

    return np / objSize;
}


ObjectVector::ObjectVector(const MirState *state, std::string name, float mass, int objSize, int nObjects) :
    ObjectVector( state, name, mass, objSize,
                  std::make_unique<LocalObjectVector>(this, objSize, nObjects),
                  std::make_unique<LocalObjectVector>(this, objSize, 0) )
{}

ObjectVector::ObjectVector(const MirState *state, std::string name, float mass, int objSize,
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

void ObjectVector::findExtentAndCOM(cudaStream_t stream, ParticleVectorType type)
{
    bool isLocal = (type == ParticleVectorType::Local);
    auto lov = isLocal ? local() : halo();

    debug("Computing COM and extent OV '%s' (%s)", name.c_str(), isLocal ? "local" : "halo");

    OVview view(this, lov);
    
    constexpr int warpSize = 32;
    const int nthreads = 128;
    const int nblocks = getNblocks(view.nObjects * warpSize, nthreads);
    
    SAFE_KERNEL_LAUNCH(
            ObjectVectorKernels::minMaxCom,
            nblocks, nthreads, 0, stream,
            view );
}

static std::vector<float3> getCom(DomainInfo domain,
                                  const PinnedBuffer<COMandExtent>& com_extents)
{
    int n = com_extents.size();
    std::vector<float3> pos(n);

    for (int i = 0; i < n; ++i) {
        auto r = com_extents[i].com;
        pos[i] = domain.local2global(r);
    }

    return pos;
}

void ObjectVector::_checkpointObjectData(MPI_Comm comm, std::string path, int checkpointId)
{
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointNameWithId(path, RestartOVIdentifier, "", checkpointId);
    info("Checkpoint for object vector '%s', writing to file %s",
         name.c_str(), filename.c_str());

    auto coms_extents = local()->dataPerObject.getData<COMandExtent>(ChannelNames::comExtents);

    coms_extents->downloadFromDevice(defaultStream, ContainersSynch::Synch);
    
    auto positions = std::make_shared<std::vector<float3>>(getCom(state->domain, *coms_extents));

    XDMF::VertexGrid grid(positions, comm);

    auto channels = CheckpointHelpers::extractShiftPersistentData(state->domain,
                                                                  local()->dataPerObject);
    
    XDMF::write(filename, &grid, channels, comm);

    createCheckpointSymlink(comm, path, RestartOVIdentifier, "xmf", checkpointId);

    debug("Checkpoint for object vector '%s' successfully written", name.c_str());
}

void ObjectVector::_restartObjectData(MPI_Comm comm, std::string path,
                                      const ObjectVector::ExchMapSize& ms)
{
    constexpr int objChunkSize = 1; // only one datum per object
    CUDA_Check( cudaDeviceSynchronize() );

    auto filename = createCheckpointName(path, RestartOVIdentifier, "xmf");
    info("Restarting object vector %s from file %s", name.c_str(), filename.c_str());

    auto listData = RestartHelpers::readData(filename, comm, objChunkSize);

    // remove positions from the read data (artificial for non rov)
    RestartHelpers::extractChannel<float3> (ChannelNames::XDMF::position, listData);
    
    RestartHelpers::exchangeListData(comm, ms.map, listData, objChunkSize);
    RestartHelpers::requireExtraDataPerObject(listData, this);

    auto& dataPerObject = local()->dataPerObject;
    dataPerObject.resize_anew(ms.newSize);

    RestartHelpers::copyAndShiftListData(state->domain, listData, dataPerObject);
    
    info("Successfully read object infos of '%s'", name.c_str());
}

void ObjectVector::checkpoint(MPI_Comm comm, std::string path, int checkpointId)
{
    _checkpointParticleData(comm, path, checkpointId);
    _checkpointObjectData  (comm, path, checkpointId);
}

void ObjectVector::restart(MPI_Comm comm, std::string path)
{
    auto ms = _restartParticleData(comm, path, objSize);
    _restartObjectData(comm, path, ms);
    
    local()->resize(ms.newSize * objSize, defaultStream);
}
