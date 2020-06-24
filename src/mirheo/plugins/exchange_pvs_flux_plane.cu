// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "exchange_pvs_flux_plane.h"

#include <mirheo/core/pvs/packers/particles.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace exchange_pvs_flux_plane_kernels
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

} // namespace exchange_pvs_flux_plane_kernels


ExchangePVSFluxPlanePlugin::ExchangePVSFluxPlanePlugin(const MirState *state, std::string name, std::string pv1Name, std::string pv2Name, real4 plane) :
    SimulationPlugin(state, name),
    pv1Name_(pv1Name),
    pv2Name_(pv2Name),
    plane_(plane),
    numberCrossedParticles_(1)
{
    // we will copy positions and velocities manually in the kernel
    PackPredicate predicate = [](const DataManager::NamedChannelDesc& namedDesc)
    {
        auto channelName = namedDesc.first;
        auto channelDesc = namedDesc.second;
        return
            (channelName != channel_names::positions) &&
            (channelName != channel_names::velocities) &&
            (channelDesc->persistence == DataManager::PersistenceMode::Active);
    };

    extra1_ = std::make_unique<ParticlePacker>(predicate);
    extra2_ = std::make_unique<ParticlePacker>(predicate);
}

ExchangePVSFluxPlanePlugin::~ExchangePVSFluxPlanePlugin() = default;

void ExchangePVSFluxPlanePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv1_ = simulation->getPVbyNameOrDie(pv1Name_);
    pv2_ = simulation->getPVbyNameOrDie(pv2Name_);

    pv1_->requireDataPerParticle<real4> (channel_names::oldPositions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
    pv2_->requireDataPerParticle<real4> (channel_names::oldPositions, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
}

void ExchangePVSFluxPlanePlugin::beforeCellLists(cudaStream_t stream)
{
    const DomainInfo domain = getState()->domain;
    PVviewWithOldParticles view1(pv1_, pv1_->local());
    PVview                 view2(pv2_, pv2_->local());
    const int nthreads = 128;

    numberCrossedParticles_.clear(stream);

    SAFE_KERNEL_LAUNCH(
            exchange_pvs_flux_plane_kernels::countParticles,
            getNblocks(view1.size, nthreads), nthreads, 0, stream,
            domain, view1, plane_, numberCrossedParticles_.devPtr() );

    numberCrossedParticles_.downloadFromDevice(stream, ContainersSynch::Synch);

    const int numPartsExchange = numberCrossedParticles_[0];
    const int old_size2 = view2.size;
    const int new_size2 = old_size2 + numPartsExchange;

    pv2_->local()->resize(new_size2, stream);
    numberCrossedParticles_.clear(stream);

    view2 = PVview(pv2_, pv2_->local());

    extra1_->update(pv1_->local(), stream);
    extra2_->update(pv2_->local(), stream);

    SAFE_KERNEL_LAUNCH(
        exchange_pvs_flux_plane_kernels::moveParticles,
        getNblocks(view1.size, nthreads), nthreads, 0, stream,
        domain, view1, view2, plane_, old_size2, numberCrossedParticles_.devPtr(),
        extra1_->handler(), extra2_->handler() );

    if (numPartsExchange > 0)
    {
        pv1_->cellListStamp++;
        pv2_->cellListStamp++;
    }
}

} // namespace mirheo
