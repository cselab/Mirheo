// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "sinusoidal_field.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace sinusoidal_field_kernels {

__global__ void computeField(DomainInfo domain, PVview view,
                             real magnitude, int waveNumber,
                             real4 *dst)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= view.size)
        return;

    const real3 pos = Real3_int(view.readPosition(i)).v;
    const real3 r = domain.local2global(pos);

    const real k = waveNumber * 2 * M_PI / domain.globalSize.y;
    const real4 v {magnitude * math::sin(k * r.y),
                   0.0_r, 0.0_r, 0.0_r};

    dst[i] = v;
}

} // namespace sinusoidal_field_kernels

SinusoidalFieldPlugin::SinusoidalFieldPlugin(const MirState *state, std::string name, std::string pvName,
                                             real magnitude, int waveNumber, std::string sfChannelName) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    pv_(nullptr),
    magnitude_(magnitude),
    waveNumber_(waveNumber),
    sfChannelName_(sfChannelName)
{
    channel_names::failIfReserved(sfChannelName_, channel_names::reservedParticleFields);
}

void SinusoidalFieldPlugin::beforeCellLists(cudaStream_t stream)
{
    auto& dataManager = pv_->local()->dataPerParticle;
    PinnedBuffer<real4> *sf = dataManager.getData<real4>(sfChannelName_);

    const auto domain = getState()->domain;

    const PVview view(pv_, pv_->local());

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    SAFE_KERNEL_LAUNCH(
        sinusoidal_field_kernels::computeField,
        nblocks, nthreads, 0, stream,
        domain, view,
        magnitude_, waveNumber_,
        sf->devPtr());
}

bool SinusoidalFieldPlugin::needPostproc()
{
    return false;
}

void SinusoidalFieldPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    pv_->requireDataPerParticle<real4>(sfChannelName_, DataManager::PersistenceMode::Active);
}

} // namespace mirheo
