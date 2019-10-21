#include "particle_portal.h"

#include <core/pvs/object_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace ParticlePortal {

// (*) In the code below, we don't manually check whether particles are marked
// or not because marked particles are always outside of the box. The following
// `static_assert` is to ensure the compilation fails if this changes.
static_assert(Float3_int::mark_val < -100.f,
              "The assumption that marked particles are far outside of "
              "the simulation domain is not correct anymore?");

static bool packPredicate(const DataManager::NamedChannelDesc& namedDesc) noexcept {
    return namedDesc.second->persistence == DataManager::PersistenceMode::Active;
};

static __device__ bool isPointInsideBox(const float3 &point, const float3 &lo, const float3 &hi)
{
    return lo.x <= point.x && point.x < hi.x
        && lo.y <= point.y && point.y < hi.y
        && lo.z <= point.z && point.z < hi.z;
}

__global__ void countParticlesInBox(
        PVview view,
        float3 lo,  // Local coordinate system.
        float3 hi,
        int *numParticlesToSend)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    if (isPointInsideBox(make_float3(view.readPosition(pid)), lo, hi))  // (*)
        atomicAggInc(numParticlesToSend);
}

__global__ void packParticles(
        PVview view,
        ParticlePackerHandler handler,
        float3 lo,  // Local coordinate system.
        float3 hi,
        float3 shift,
        int numParticlesToSend,
        char *dstBuffer,
        int *counter)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    if (isPointInsideBox(make_float3(view.readPosition(pid)), lo, hi))  // (*)
    {
        // The shift includes the local->global domain transformation.
        int dst = atomicAggInc(counter);
        handler.particles.packShift(pid, dst, dstBuffer, numParticlesToSend, shift);
    }
}

__global__ void removeParticlesInBox(
        PVview view,
        float3 lo,  // Local coordinate system.
        float3 hi)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p = view.readParticle(pid);
    if (isPointInsideBox(p.r, lo, hi))  // (*)
    {
        p.mark();
        view.writeParticle(pid, p);
    }
}

__global__ void unpackParticles(
        PVview view,
        ParticlePackerHandler handler,
        float3 shift,
        const char *srcBuffer,
        int numExisting,
        int numNew)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNew) return;

    handler.particles.unpackShift(i, numExisting + i, srcBuffer, numNew, shift);
}

} // namespace ParticlePortal


ParticlePortalCommon::ParticlePortalCommon(
        const MirState *state, std::string name, std::string pvName,
        float3 position, float3 size, int tag, MPI_Comm interCommExternal) :
    SimulationPlugin(state, name),
    pvName(pvName),
    localLo(state->domain.global2local(position)),
    localHi(state->domain.global2local(position + size)),
    tag(tag),
    interCommExternal(interCommExternal),
    packer(ParticlePortal::packPredicate)
{
    int flag;
    MPI_Check( MPI_Comm_test_inter(interCommExternal, &flag) );
    if (!flag)
        throw std::invalid_argument("Expected an intercommunicator, got an intracommunicator.");
}

ParticlePortalCommon::~ParticlePortalCommon() = default;

void ParticlePortalCommon::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
}


ParticlePortalSource::ParticlePortalSource(
        const MirState *state, std::string name, std::string pvName,
        float3 src, float3 dst, float3 size, int tag, MPI_Comm interCommExternal) :
    ParticlePortalCommon(state, name, pvName, src, size, tag, interCommExternal),
    shift(state->domain.local2global(dst - src))  // (src local --> src global shift)
                                                  // + (src global --> dst global)
{}

ParticlePortalSource::~ParticlePortalSource() = default;


void ParticlePortalSource::beforeCellLists(cudaStream_t stream)
{
    const int nthreads = 128;

    PVview view(pv, pv->local());

    // 1. Count how many particles to send.
    numParticlesToSend.clearDevice(stream);
    SAFE_KERNEL_LAUNCH(
        ParticlePortal::countParticlesInBox,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, localLo, localHi, numParticlesToSend.devPtr());

    // 2. Allocate a temporary buffer, copy and download.
    numParticlesToSend.downloadFromDevice(stream);
    packer.update(pv->local(), stream);

    int numSend   = numParticlesToSend[0];
    int sizeBytes = (int)packer.getSizeBytes(numSend);
    outBuffer.resize_anew(sizeBytes);

    SAFE_KERNEL_LAUNCH(
        ParticlePortal::packParticles,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, packer.handler(), localLo, localHi, shift,
        numSend, outBuffer.devPtr(), numParticlesToSend.devPtr() + 1);

    // 3. Send. This should be async...
    outBuffer.downloadFromDevice(stream);

    int size[2] = {numSend, sizeBytes};
    MPI_Request request[2];
    MPI_Check( MPI_Isend(&size, 2, MPI_INT, 0, tag, interCommExternal, &request[0]) );
    MPI_Check( MPI_Isend(outBuffer.hostPtr(), sizeBytes, MPI_BYTE, 0, tag, interCommExternal, &request[1]) );
    MPI_Check( MPI_Waitall(2, request, MPI_STATUSES_IGNORE) );
}



ParticlePortalDestination::ParticlePortalDestination(
        const MirState *state, std::string name, std::string pvName,
        __UNUSED float3 src, float3 dst, float3 size, int tag, MPI_Comm interCommExternal) :
    ParticlePortalCommon(state, name, pvName, dst, size, tag, interCommExternal),
    shift(state->domain.global2local(float3{0.0, 0.0, 0.0}))  // (dst global -> dst local)
{}

ParticlePortalDestination::~ParticlePortalDestination() = default;


void ParticlePortalDestination::beforeCellLists(cudaStream_t stream)
{
    const int nthreads = 128;

    // 1. Receive data over MPI, allocate buffers and upload to GPU.
    int recv[2];  // Number of particles and size in bytes (as a check).
    MPI_Check( MPI_Recv(&recv, 2, MPI_INT, 0, tag, interCommExternal, MPI_STATUS_IGNORE) );

    packer.update(pv->local(), stream);  // Update to compute size.

    int numRecv = recv[0];  // Number of incoming objects.
    int sizeBytes = (int)packer.getSizeBytes(numRecv);
    assert(sizeBytes == recv[1]);
    inBuffer.resize_anew(sizeBytes);

    MPI_Check( MPI_Recv(inBuffer.hostPtr(), sizeBytes, MPI_BYTE, 0, tag, interCommExternal, MPI_STATUS_IGNORE) );
    inBuffer.uploadToDevice(stream);

    // 2. Remove old particles.
    PVview view(pv, pv->local());
    SAFE_KERNEL_LAUNCH(
        ParticlePortal::removeParticlesInBox,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, localLo, localHi);

    // 3. Resize the particle vector and unpack them.
    int numExisting = pv->local()->size();
    pv->local()->resize(numExisting + numRecv, stream);
    view = PVview(pv, pv->local());
    packer.update(pv->local(), stream);  // Update again to get the new PV buffers.

    SAFE_KERNEL_LAUNCH(
        ParticlePortal::unpackParticles,
        getNblocks(numRecv, nthreads), nthreads, 0, stream,
        view, packer.handler(), shift, inBuffer.devPtr(), numExisting, numRecv);
}
