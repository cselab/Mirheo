// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "displacement.h"
#include "utils/time_stamp.h"

#include <mirheo/core/simulation.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace particle_displacement_plugin_kernels
{

__global__ void extractPositions(PVview view, real4 *positions)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > view.size) return;
    positions[i] = view.readPosition(i);
}

__global__ void computeDisplacementsAndSavePositions(PVview view, real4 *positions, real3 *displacements)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > view.size) return;

    Real3_int pos(view.readPosition(i));
    Real3_int oldPos(positions[i]);

    displacements[i] = pos.v - oldPos.v;
    positions[i] = pos.toReal4();
}

} // namespace particle_displacement_plugin_kernels

const std::string ParticleDisplacementPlugin::displacementChannelName_ = "displacements";
const std::string ParticleDisplacementPlugin::savedPositionChannelName_ = "saved_positions_displacements";


ParticleDisplacementPlugin::ParticleDisplacementPlugin(const MirState *state, std::string name, std::string pvName, int updateEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    pv_(nullptr),
    updateEvery_(updateEvery)
{}

ParticleDisplacementPlugin::~ParticleDisplacementPlugin() = default;

void ParticleDisplacementPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    channel_names::failIfReserved(displacementChannelName_,  channel_names::reservedParticleFields);
    channel_names::failIfReserved(savedPositionChannelName_, channel_names::reservedParticleFields);

    pv_->requireDataPerParticle<real3>(displacementChannelName_,
                                      DataManager::PersistenceMode::Active);

    pv_->requireDataPerParticle<real4>(savedPositionChannelName_,
                                      DataManager::PersistenceMode::Active,
                                      DataManager::ShiftMode::Active);

    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    auto& manager      = pv_->local()->dataPerParticle;
    auto positions     = manager.getData<real4>(savedPositionChannelName_);
    auto displacements = manager.getData<real3>(displacementChannelName_);

    displacements->clear(defaultStream);

    SAFE_KERNEL_LAUNCH(
            particle_displacement_plugin_kernels::extractPositions,
            getNblocks(view.size, nthreads), nthreads, 0, defaultStream,
            view, positions->devPtr());
}

void ParticleDisplacementPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), updateEvery_)) return;

    auto& manager = pv_->local()->dataPerParticle;

    auto positions     = manager.getData<real4>(savedPositionChannelName_);
    auto displacements = manager.getData<real3>( displacementChannelName_);

    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            particle_displacement_plugin_kernels::computeDisplacementsAndSavePositions,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, positions->devPtr(), displacements->devPtr());
}

} // namespace mirheo
