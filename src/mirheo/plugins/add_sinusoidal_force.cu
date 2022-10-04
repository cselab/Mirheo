// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "add_sinusoidal_force.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace add_sin_force_kernels {

__global__ void addForce(PVview view, DomainInfo domain,
                         real magnitude, int waveNumber)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.size) return;

    const real3 rLocal = Real3_int(view.readPosition(i)).v;
    const real3 r = domain.local2global(rLocal);
    const real k = 2.0 * M_PI / domain.globalSize.y;

    const real3 force {magnitude * math::sin(waveNumber * r.y * k),
                       0.0_r,
                       0.0_r};

    view.forces[i] += make_real4(force, 0.0_r);
}

} // namespace add_sin_force_kernels

AddSinusoidalForcePlugin::AddSinusoidalForcePlugin(const MirState *state,
                                                   const std::string& name,
                                                   const std::string& pvName,
                                                   real magnitude,
                                                   int waveNumber) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    magnitude_(magnitude),
    waveNumber_(waveNumber)
{}

void AddSinusoidalForcePlugin::setup(Simulation *simulation, const MPI_Comm& comm,
                                     const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);
}

void AddSinusoidalForcePlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    const auto domain = getState()->domain;

    SAFE_KERNEL_LAUNCH(
        add_sin_force_kernels::addForce,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, domain, magnitude_, waveNumber_);
}

} // namespace mirheo
