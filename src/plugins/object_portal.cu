#include "object_portal.h"

#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>
#include <core/utils/kernel_launch.h>

namespace ObjectPortal {

static bool packPredicate(const DataManager::NamedChannelDesc& namedDesc) noexcept {
    // Positions are not Active, so we accept them explicitly.
    return namedDesc.second->persistence == DataManager::PersistenceMode::Active
        || namedDesc.first == ChannelNames::positions;
};

static __device__ bool areBoxesIntersecting(const float3 &lo1, const float3 &hi1,
                                            const float3 &lo2, const float3 &hi2)
{
    return !(hi1.x < lo2.x || hi2.x < lo1.x
          || hi1.y < lo2.y || hi2.y < lo2.y
          || hi1.z < lo2.z || hi2.z < lo2.z);
}

/// Check how many objects crossed the plane.
/// Also, check how many COMs are inside the portal.
__global__ void updateUUIDsAndCountInBox(
        OVview view,
        float4 plane,          // Plane. Local coordinate system.
        float *oldSides,       // Plane.
        int64_t *localUUIDs,   // Plane.
        int64_t *uuidCounter,  // Plane.
        float3 lo,             // Portal. Local coordinate system.
        float3 hi,             // Portal.
        int64_t *outUUIDs,     // Portal.
        int *outIdx,           // Portal.
        int *numObjectsToSend) // Portal.
{
    int oid = blockIdx.x * blockDim.x + threadIdx.x;
    if (oid >= view.nObjects) return;

    // Check if the object crossed the plane.
    auto &info = view.comAndExtents[oid];
    float newSide = plane.x * info.com.x + plane.y * info.com.y + plane.z * info.com.z + plane.w;
    if (oldSides[oid] < 0.f && newSide >= 0.f)
        localUUIDs[oid] = atomicAdd(uuidCounter, 1);

    // Update always. It's possible that oldSides > 0 and newSide < 0 due to
    // periodic boundary conditions.
    oldSides[oid] = newSide;

    // Check if the object's bounding box intersects the portal.
    if (areBoxesIntersecting(info.low, info.high, lo, hi)) {
        int id = atomicAdd(numObjectsToSend, 1);
        outIdx[id] = oid;
        outUUIDs[id] = localUUIDs[oid];
    }
}

__global__ void packObjects(
        OVview view,
        ObjectPackerHandler handler,
        float3 shift,
        const int *outIdx,
        int numObjectsToSend,
        char *outBuffer)
{
    int oid = blockIdx.x;  // Object.
    // if (oid >= numObjectsToSend) return;  // Already true, no need to check.

    handler.blockPackShift(numObjectsToSend, outBuffer, outIdx[oid], oid, shift);
}

/// Match UUIDs of local objects with those of incoming.
/// Stores the number of matches in `counters` and (incoming id, local id) pairs in `indexPairs`.
__global__ void matchUUIDs(
        const int64_t *localUUIDs,
        int nObjects,
        const int64_t *inUUIDs,
        bool *visited,
        int *indexPairs,
        int numRecv,
        int *counters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = i / numRecv;  // Local (existing) object index.
    int iid = i % numRecv;  // Incoming object index.
    if (lid >= nObjects) return;

    if (localUUIDs[lid] == inUUIDs[iid]) {
        int idx = atomicAdd(&counters[0], 1);
        atomicAdd(&counters[1], 1);
        indexPairs[2 * idx]     = iid;  // Source index (incoming buffer).
        indexPairs[2 * idx + 1] = lid;  // Destination index (object vector).
        visited[iid] = 1;
    }
}

__global__ void findNew(
        const bool *visited,
        int numObjects,
        int numRecv,
        int *indexPairs,
        int *counter)
{
    int iid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iid >= numRecv) return;

    if (!visited[iid]) {
        int cnt = atomicAdd(counter, 1);             // Number of added objects.
        indexPairs[2 * cnt]     = iid;               // Source index (incoming buffer).
        indexPairs[2 * cnt + 1] = numObjects + cnt;  // Destination index (object vector).
    }
}

__global__ void unpackObjects(
        OVview view,
        ObjectPackerHandler handler,
        float3 shift,
        const int64_t *inUUIDs,
        const char *inBuffer,
        const int *indexPairs,
        int numRecv,
        int64_t *localUUIDs)
{
    int i = blockIdx.x;
    // if (i >= numRecv) return;  // Already true, no need to check.

    int srcIdx = indexPairs[2 * i];
    int dstIdx = indexPairs[2 * i + 1];
    handler.blockUnpackShift(numRecv, inBuffer, srcIdx, dstIdx, shift);
    localUUIDs[dstIdx] = inUUIDs[i];
}

} // namespace ObjectPortal


ObjectPortalCommon::ObjectPortalCommon(
        const MirState *state, std::string name, std::string ovName,
        float3 position, float3 size, int tag, MPI_Comm interCommExternal) :
    SimulationPlugin(state, name),
    ovName(ovName),
    uuidChannelName(name + "_UUID"),
    localLo(state->domain.global2local(position)),
    localHi(state->domain.global2local(position + size)),
    tag(tag),
    interCommExternal(interCommExternal),
    packer(ObjectPortal::packPredicate)
{
    int flag;
    MPI_Check( MPI_Comm_test_inter(interCommExternal, &flag) );
    if (!flag)
        throw std::invalid_argument("Expected an intercommunicator, got an intracommunicator.");
}

ObjectPortalCommon::~ObjectPortalCommon() = default;

void ObjectPortalCommon::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    // Set UUIDs as non-active because we send them separately.
    ov = simulation->getOVbyNameOrDie(ovName);
    ov->requireDataPerObject<int64_t>(uuidChannelName, DataManager::PersistenceMode::None);
}


ObjectPortalSource::ObjectPortalSource(
        const MirState *state, std::string name, std::string ovName,
        float3 src, float3 dst, float3 size, float4 plane, int tag, MPI_Comm interCommExternal) :
    ObjectPortalCommon(state, name, ovName, src, size, tag, interCommExternal),
    oldSideChannelName(name + "_oldSide"),
    plane(state->domain.global2localPlane(plane)),
    shift(state->domain.local2global(dst - src))  // (src local --> src global shift)
                                                  // + (src global --> dst global)
{
}

ObjectPortalSource::~ObjectPortalSource() = default;

void ObjectPortalSource::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    ObjectPortalCommon::setup(simulation, comm, interComm);

    ov->requireDataPerObject<float>(oldSideChannelName, DataManager::PersistenceMode::None);

    auto& manager    = ov->local()->dataPerObject;
    auto& localUUIDs = *manager.getData<int64_t>(uuidChannelName);
    auto& oldSides   = *manager.getData<float>(oldSideChannelName);
    localUUIDs.resize_anew(ov->local()->nObjects);
    oldSides  .resize_anew(ov->local()->nObjects);

    // UUIDs are initially equal to globalIDs. We assume here that globalIDs
    // are unique.
    if (simulation->nranks3D.x != 1 || simulation->nranks3D.y != 1 || simulation->nranks3D.z != 1) {
        // For multiple ranks, use a large offset for UUIDs (e.g. rank * 1e9).
        throw std::logic_error("Multiple ranks not implemented yet. UUIDs and exchange would be broken.");
    }
    localUUIDs.copy(*manager.getData<int64_t>(ChannelNames::globalIds), defaultStream);
    oldSides.clearDevice(defaultStream);
}

void ObjectPortalSource::afterIntegration(cudaStream_t stream)
{
    const int nthreads = 128;

    auto& manager    = ov->local()->dataPerObject;
    auto& localUUIDs = *manager.getData<int64_t>(uuidChannelName);
    auto& oldSides   = *manager.getData<float>(oldSideChannelName);

    // 1a. Update COMs and UUIDs -- check if objects crossed the plane.
    // 1b. Find which objects overlap the portal box.
    ov->findExtentAndCOM(stream, ParticleVectorType::Local);

    OVview view(ov, ov->local());
    numObjectsToSend.clearDevice(stream);
    outIdx  .resize_anew(view.nObjects);  // Indices of the object to send.
    outUUIDs.resize_anew(view.nObjects);

    SAFE_KERNEL_LAUNCH(
        ObjectPortal::updateUUIDsAndCountInBox,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, plane, oldSides.devPtr(), localUUIDs.devPtr(), uuidCounter.devPtr(),
        localLo, localHi, outUUIDs.devPtr(), outIdx.devPtr(), numObjectsToSend.devPtr());

    // 2. Filter particles inside the portal box.
    numObjectsToSend.downloadFromDevice(stream);
    packer.update(ov->local(), stream);

    int numSend = numObjectsToSend[0];
    int sizeBytes = (int)packer.getSizeBytes(numSend);
    outBuffer.resize_anew(sizeBytes);

    SAFE_KERNEL_LAUNCH(
        ObjectPortal::packObjects,
        numSend, nthreads, 0, stream,
        view, packer.handler(), shift, outIdx.devPtr(), numSend, outBuffer.devPtr());

    // 3. Send. This should be async...
    outUUIDs.resize_anew(numSend);  // Shrink before download.
    outUUIDs.downloadFromDevice(stream, ContainersSynch::Asynch);
    outBuffer.downloadFromDevice(stream);

    int size[2] = {numSend, sizeBytes};
    MPI_Request request[3];
    MPI_Check( MPI_Isend(&size, 2, MPI_INT, 0, tag, interCommExternal, &request[0]) );
    MPI_Check( MPI_Isend(outUUIDs.hostPtr(), numSend, MPI_INT64_T, 0, tag, interCommExternal, &request[1]) );
    MPI_Check( MPI_Isend(outBuffer.hostPtr(), sizeBytes, MPI_BYTE, 0, tag, interCommExternal, &request[1]) );
    MPI_Check( MPI_Waitall(2, request, MPI_STATUSES_IGNORE) );
}


ObjectPortalDestination::ObjectPortalDestination(
        const MirState *state, std::string name, std::string ovName,
        __UNUSED float3 src, float3 dst, float3 size, int tag, MPI_Comm interCommExternal) :
    ObjectPortalCommon(state, name, ovName, dst, size, tag, interCommExternal),
    shift(state->domain.global2local(float3{0.f, 0.f, 0.f}))  // (dst global -> dst local)
{}

ObjectPortalDestination::~ObjectPortalDestination() = default;


void ObjectPortalDestination::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    ObjectPortalCommon::setup(simulation, comm, interComm);

    if (ov->local()->nObjects != 0) {
        // Maybe start with a large UUID offset?
        // This would solve the rank (UUID-related) issue as well.
        throw std::logic_error(
                "Start with non-zero number of objects on the destination side not "
                "implemented due to the potential overlap of UUIDs with "
                "those of incoming objects sent by the portal source.");
    }
}



void ObjectPortalDestination::afterIntegration(cudaStream_t stream)
{
    const int nthreads = 128;

    // 1. Receive data over MPI, allocate buffers and upload to GPU.
    int recv[2];  // Number of particles and size in bytes (as a check).
    MPI_Check( MPI_Recv(&recv, 2, MPI_INT, 0, tag, interCommExternal, MPI_STATUS_IGNORE) );

    packer.update(ov->local(), stream);  // Update to compute size.

    int numRecv = recv[0];  // Number of incoming objects.
    int sizeBytes = (int)packer.getSizeBytes(numRecv);
    if (sizeBytes != recv[1]) {
        throw std::runtime_error("Expected package size does not match received match size. "
                                 "Sender and receiver likely have inconsistent channels.");
    }
    inUUIDs.resize_anew(numRecv);
    inBuffer.resize_anew(sizeBytes);

    MPI_Request requests[2];
    MPI_Check( MPI_Irecv(inUUIDs.hostPtr(), numRecv, MPI_INT64_T, 0, tag, interCommExternal, &requests[0]) );
    MPI_Check( MPI_Irecv(inBuffer.hostPtr(), sizeBytes, MPI_BYTE, 0, tag, interCommExternal, &requests[1]) );

    indexPairs.resize_anew(2 * numRecv);
    visited.resize_anew(numRecv);
    visited.clearDevice(stream);

    MPI_Check( MPI_Waitall(2, requests, MPI_STATUSES_IGNORE) );
    if (!numRecv)
        return;

    inUUIDs.uploadToDevice(stream);
    inBuffer.uploadToDevice(stream);

    // 2. Find which incoming objects match existing ones, which are new.
    auto& manager    = ov->local()->dataPerObject;
    auto& localUUIDs = *manager.getData<int64_t>(uuidChannelName);
    OVview view(ov, ov->local());

    counters.clear(stream);
    SAFE_KERNEL_LAUNCH(
        ObjectPortal::matchUUIDs,
        getNblocks(numRecv * view.nObjects, nthreads), nthreads, 0, stream,
        localUUIDs.devPtr(), view.nObjects, inUUIDs.devPtr(),
        visited.devPtr(), indexPairs.devPtr(), numRecv, counters.devPtr());

    SAFE_KERNEL_LAUNCH(
        ObjectPortal::findNew,
        getNblocks(numRecv, nthreads), nthreads, 0, stream,
        visited.devPtr(), view.nObjects, numRecv,
        indexPairs.devPtr(), counters.devPtr() + 1);

    counters.downloadFromDevice(stream);
    int toOverwrite = counters[0];
    int toInsert    = counters[1] - counters[0];
    assert(counters[1] == numRecv);

    // 3. Copy all incoming objects to the obstacle vector.
    ov->local()->resize(view.size + toInsert * view.objSize, stream);
    view = OVview(ov, ov->local());
    packer.update(ov->local(), stream);  // Update again to get the new OV buffers.

    SAFE_KERNEL_LAUNCH(
        ObjectPortal::unpackObjects,
        numRecv, nthreads, 0, stream,
        view, packer.handler(), shift,
        inUUIDs.devPtr(), inBuffer.devPtr(), indexPairs.devPtr(), numRecv, localUUIDs.devPtr());
}
