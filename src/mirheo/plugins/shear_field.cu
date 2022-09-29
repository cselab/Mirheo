// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "shear_field.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace shear_field_kernels {

__global__ void computeShear(DomainInfo domain, PVview view,
                             real3 shearx, real3 sheary, real3 shearz,
                             real3 origin, real4 *dst)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= view.size)
        return;

    const real3 pos = Real3_int(view.readPosition(i)).v;
    const real3 r = domain.local2global(pos) - origin;

    const real4 v {dot(shearx, r),
                   dot(sheary, r),
                   dot(shearz, r),
                   0.0_r};

    dst[i] = v;
}

} // namespace shear_field_kernels

ShearFieldPlugin::ShearFieldPlugin(const MirState *state, std::string name, std::string pvName,
                                   std::array<real,9> shear, real3 origin, std::string sfChannelName) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    pv_(nullptr),
    shear_(shear),
    origin_(origin),
    sfChannelName_(sfChannelName)
{
    channel_names::failIfReserved(sfChannelName_, channel_names::reservedParticleFields);
}

void ShearFieldPlugin::beforeCellLists(cudaStream_t stream)
{
    auto& dataManager = pv_->local()->dataPerParticle;
    PinnedBuffer<real4> *sf = dataManager.getData<real4>(sfChannelName_);

    const auto domain = getState()->domain;
    const real3 shearx {shear_[0], shear_[1], shear_[2]};
    const real3 sheary {shear_[3], shear_[4], shear_[5]};
    const real3 shearz {shear_[6], shear_[7], shear_[8]};

    const PVview view(pv_, pv_->local());

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    SAFE_KERNEL_LAUNCH(
        shear_field_kernels::computeShear,
        nblocks, nthreads, 0, stream,
        domain, view,
        shearx, sheary, shearz,
        origin_, sf->devPtr());
}

bool ShearFieldPlugin::needPostproc()
{
    return false;
}

void ShearFieldPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    pv_->requireDataPerParticle<real4>(sfChannelName_, DataManager::PersistenceMode::Active);
}

} // namespace mirheo
