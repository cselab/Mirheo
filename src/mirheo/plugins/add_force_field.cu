// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "add_force_field.h"

#include <mirheo/core/field/from_file.h>
#include <mirheo/core/field/from_function.h>
#include <mirheo/core/field/utils.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace add_force_field_kernels {

__global__ void addForce(PVview view, VectorFieldDeviceHandler forceField)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= view.size)
        return;

    const real3 r = Real3_int(view.readPosition(i)).v;

    view.forces[i] += forceField(r);
}

} // namespace add_force_field_kernels

constexpr real3 defaultMargin {5.0_r, 5.0_r, 5.0_r};

AddForceFieldPlugin::AddForceFieldPlugin(const MirState *state,
                                         const std::string& name,
                                         const std::string& pvName,
                                         ForceField forceField,
                                         real3 gridSpacing)
    : SimulationPlugin(state, name)
    , pvName_(pvName)
    , forceField_(std::make_unique<VectorFieldFromFunction>
                  (state, name+"_field", std::move(forceField), gridSpacing, defaultMargin))
{}

AddForceFieldPlugin::AddForceFieldPlugin(const MirState *state,
                                         const std::string& name,
                                         const std::string& pvName,
                                         std::string forceFieldFilename,
                                         real3 gridSpacing)
    : SimulationPlugin(state, name)
    , pvName_(pvName)
    , forceField_(std::make_unique<VectorFieldFromFile>
                  (state, name+"_field", std::move(forceFieldFilename),
                   gridSpacing, defaultMargin))
{}

void AddForceFieldPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    forceField_->setup(comm);
}

void AddForceFieldPlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        add_force_field_kernels::addForce,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, forceField_->handler() );
}


} // namespace mirheo
