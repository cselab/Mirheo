// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "add_pressure_gradient.h"

#include <mirheo/core/field/from_function.h>
#include <mirheo/core/field/utils.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace add_pressure_gradients_kernels {

__global__ void addPressureGradient(PVview view, ScalarFieldDeviceHandler pressureField)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= view.size)
        return;

    const real3 r = Real3_int(view.readPosition(i)).v;

    constexpr real h = 0.05_r;
    const real3 grad = computeGradient(pressureField, r, h);

    view.forces[i] += make_real4(-grad, 0.0_r);
}

} // namespace add_pressure_gradients_kernels

constexpr real3 defaultMargin {5.0_r, 5.0_r, 5.0_r};

AddPressureGradientPlugin::AddPressureGradientPlugin(const MirState *state,
                                                     const std::string& name,
                                                     const std::string& pvName,
                                                     PressureField pressureField,
                                                     real3 gridSpacing)
    : SimulationPlugin(state, name)
    , pvName_(pvName)
    , pressureField_(std::make_unique<ScalarFieldFromFunction>
                     (state, name+"_field", pressureField, gridSpacing, defaultMargin))
{}

void AddPressureGradientPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    pressureField_->setup(comm);
}

void AddPressureGradientPlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        add_pressure_gradients_kernels::addPressureGradient,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, pressureField_->handler() );
}


} // namespace mirheo
