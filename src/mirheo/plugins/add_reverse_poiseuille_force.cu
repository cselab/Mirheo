// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "add_reverse_poiseuille_force.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace add_force_kernels {

__global__ void addForce(PVview view, real3 force, DomainInfo domain, char flipDirection)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.size) return;

    const real3 rLocal = Real3_int(view.readPosition(i)).v;

    const real3 r = domain.local2global(rLocal);

    if ((flipDirection == 'x' && r.x > 0.5_r * domain.globalSize.x) ||
        (flipDirection == 'y' && r.y > 0.5_r * domain.globalSize.y) ||
        (flipDirection == 'z' && r.z > 0.5_r * domain.globalSize.z))
        force = -force;

    view.forces[i] += make_real4(force, 0.0_r);
}

} // namespace add_force_kernels

AddReversePoiseuilleForcePlugin::AddReversePoiseuilleForcePlugin(const MirState *state,
                                                                 const std::string& name,
                                                                 const std::string& pvName,
                                                                 real3 force,
                                                                 char flipDirection) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    force_(force),
    flipDirection_(flipDirection)
{
    if (flipDirection != 'x' &&
        flipDirection != 'y' &&
        flipDirection != 'z')
        die("Wrong value of flipDirection: must be x, y or z; got %c\n", flipDirection);
}

void AddReversePoiseuilleForcePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    AddReversePoiseuilleForcePlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);
}

void AddReversePoiseuilleForcePlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    const auto domain = getState()->domain;

    SAFE_KERNEL_LAUNCH(
            add_force_kernels::addForce,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, force_, domain, flipDirection_ );
}

} // namespace mirheo
