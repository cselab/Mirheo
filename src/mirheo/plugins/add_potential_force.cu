// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "add_potential_force.h"

#include <mirheo/core/field/from_file.h>
#include <mirheo/core/field/from_function.h>
#include <mirheo/core/field/utils.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace add_pressure_gradients_kernels {

__global__ void addPotentialForce(PVview view, ScalarFieldDeviceHandler potentialField)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= view.size)
        return;

    const real3 r = Real3_int(view.readPosition(i)).v;

    constexpr real h = 0.05_r;
    const real3 grad = computeGradient(potentialField, r, h);

    view.forces[i] += make_real4(-grad, 0.0_r);
}

} // namespace add_pressure_gradients_kernels

constexpr real3 defaultMargin {5.0_r, 5.0_r, 5.0_r};

AddPotentialForcePlugin::AddPotentialForcePlugin(const MirState *state,
                                                 const std::string& name,
                                                 const std::string& pvName,
                                                 PotentialField potentialField,
                                                 real3 gridSpacing)
    : SimulationPlugin(state, name)
    , pvName_(pvName)
    , potentialField_(std::make_unique<ScalarFieldFromFunction>
                      (state, name+"_field", potentialField, gridSpacing, defaultMargin))
{}

AddPotentialForcePlugin::AddPotentialForcePlugin(const MirState *state,
                                                 const std::string& name,
                                                 const std::string& pvName,
                                                 std::string potentialFieldFilename,
                                                 real3 gridSpacing)
    : SimulationPlugin(state, name)
    , pvName_(pvName)
    , potentialField_(std::make_unique<ScalarFieldFromFile>
                      (state, name+"_field", std::move(potentialFieldFilename),
                       gridSpacing, defaultMargin))
{}

void AddPotentialForcePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    potentialField_->setup(comm);
}

void AddPotentialForcePlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        add_pressure_gradients_kernels::addPotentialForce,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, potentialField_->handler() );
}


} // namespace mirheo
