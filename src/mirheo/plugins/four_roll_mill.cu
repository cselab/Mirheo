// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "four_roll_mill.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace add_four_roll_mill_force_kernels {

__global__ void addForce(PVview view, DomainInfo domain, real intensity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.size) return;

    const real4 pos = view.readPosition(i);
    const real3 r = domain.local2global(real3{pos.x, pos.y, pos.z});

    constexpr real twoPi = 2.0_r * static_cast<real>(M_PI);
    const real x = twoPi * r.x / domain.globalSize.x;
    const real y = twoPi * r.y / domain.globalSize.y;

    const real3 force {+intensity * sin(x) * cos(y),
                       -intensity * cos(x) * sin(y),
                       0};

    view.forces[i] += make_real4(force, 0.0_r);
}

} // namespace add_four_roll_mill_force_kernels

AddFourRollMillForcePlugin::AddFourRollMillForcePlugin(const MirState *state, const std::string& name,
                                                       const std::string& pvName, real intensity) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    intensity_(intensity)
{}

void AddFourRollMillForcePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);
}

void AddFourRollMillForcePlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());
    const int nthreads = 128;
    const auto domain = getState()->domain;

    SAFE_KERNEL_LAUNCH(
            add_four_roll_mill_force_kernels::addForce,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, domain, intensity_);
}

} // namespace mirheo
