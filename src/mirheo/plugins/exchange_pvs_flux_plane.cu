#include "exchange_pvs_flux_plane.h"

#include <mirheo/core/pvs/packers/particles.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace ExchangePvsFluxPlaneKernels
{

__device__ inline bool sidePlane(real4 plane, real3 r)
{
    return plane.x * r.x + plane.y * r.y + plane.z * r.z + plane.w >= 0._r;
}

__device__ inline bool hasCrossedPlane(DomainInfo domain, real3 pos, real3 oldPos, real4 plane)
{
    pos    = domain.local2global(pos);
    oldPos = domain.local2global(oldPos);
    return sidePlane(plane, pos) && !sidePlane(plane, oldPos);
}

__global__ void countParticles(DomainInfo domain, PVviewWithOldParticles view1, real4 plane, int *numberCrossed)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view1.size) return;

    Particle p;
    view1.readPosition   (p,    pid);
    auto rOld = view1.readOldPosition(pid);

    if (p.isMarked()) return;

    if (hasCrossedPlane(domain, p.r, rOld, plane))
        atomicAdd(numberCrossed, 1);
}

__global__ void moveParticles(DomainInfo domain, PVviewWithOldParticles view1, PVview view2,
                              real4 plane, int oldsize2, int *numberCrossed,
                              ParticlePackerHandler extra1, ParticlePackerHandler extra2)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view1.size) return;

    Particle p;
    view1.readPosition(p, pid);
    auto rOld = view1.readOldPosition(pid);

    if (p.isMarked()) return;
    
    if (hasCrossedPlane(domain, p.r, rOld, plane))
    {        
        int dst = atomicAdd(numberCrossed, 1);
        dst += oldsize2;

        view1.readVelocity(p, pid);
        view2.writeParticle(dst, p);

        p.mark();
        view1.writeParticle(pid, p);

        extra1.particles.copyTo(extra2.particles, pid, dst);
    }
}

} // namespace ExchangePvsFluxPlaneKernels


ExchangePVSFluxPlanePlugin::ExchangePVSFluxPlanePlugin(const MirState *state, std::string name, std::string pv1Name, std::string pv2Name, real4 plane) :
    SimulationPlugin(state, name),
    pv1Name(pv1Name),
    pv2Name(pv2Name),
    plane(plane),
    numberCrossedParticles(1)
{
    // we will copy positions and velocities manually in the kernel
    PackPredicate predicate = [](const DataManager::NamedChannelDesc& namedDesc)
    {
        auto name = namedDesc.first;
        auto desc = namedDesc.second;
        return
            (name != ChannelNames::positions) &&
            (name != ChannelNames::velocities) &&
            (desc->persistence == DataManager::PersistenceMode::Active);
    };

    extra1 = std::make_unique<ParticlePacker>(predicate);
    extra2 = std::make_unique<ParticlePacker>(predicate);
}

ExchangePVSFluxPlanePlugin::~ExchangePVSFluxPlanePlugin() = default;

void ExchangePVSFluxPlanePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv1 = simulation->getPVbyNameOrDie(pv1Name);
    pv2 = simulation->getPVbyNameOrDie(pv2Name);

    pv1->requireDataPerParticle<real4> (ChannelNames::oldPositions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
    pv2->requireDataPerParticle<real4> (ChannelNames::oldPositions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
}

void ExchangePVSFluxPlanePlugin::beforeCellLists(cudaStream_t stream)
{
    const DomainInfo domain = getState()->domain;
    PVviewWithOldParticles view1(pv1, pv1->local());
    PVview                 view2(pv2, pv2->local());
    const int nthreads = 128;

    numberCrossedParticles.clear(stream);
    
    SAFE_KERNEL_LAUNCH(
            ExchangePvsFluxPlaneKernels::countParticles,
            getNblocks(view1.size, nthreads), nthreads, 0, stream,
            domain, view1, plane, numberCrossedParticles.devPtr() );

    numberCrossedParticles.downloadFromDevice(stream, ContainersSynch::Synch);

    const int numPartsExchange = numberCrossedParticles[0];
    const int old_size2 = view2.size;
    const int new_size2 = old_size2 + numPartsExchange;

    pv2->local()->resize(new_size2, stream);
    numberCrossedParticles.clear(stream);

    view2 = PVview(pv2, pv2->local());    
    
    extra1->update(pv1->local(), stream);
    extra2->update(pv2->local(), stream);

    SAFE_KERNEL_LAUNCH(
        ExchangePvsFluxPlaneKernels::moveParticles,
        getNblocks(view1.size, nthreads), nthreads, 0, stream,
        domain, view1, view2, plane, old_size2, numberCrossedParticles.devPtr(),
        extra1->handler(), extra2->handler() );

    if (numPartsExchange > 0)
    {
        pv1->cellListStamp++;
        pv2->cellListStamp++;
    }
}

} // namespace mirheo
