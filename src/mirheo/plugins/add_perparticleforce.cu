// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "add_perparticleforce.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace add_per_particle_force_kernels
{

__global__ void addForce(PVview view, real3* force)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    view.forces[gid] += make_real4(force[gid], 0.0_r);
}

} // namespace add_per_particle_force_kernels

AddPerParticleForcePlugin::AddPerParticleForcePlugin(const MirState *state, const std::string& name, const std::string& pvName, const std::string& channel_name) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    channel_name_(channel_name)
{}

void AddPerParticleForcePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);
}

void AddPerParticleForcePlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            add_per_particle_force_kernels::addForce,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, pv_->local()->dataPerParticle.getData<real3>(channel_name_)->devPtr());
}

} // namespace mirheo
