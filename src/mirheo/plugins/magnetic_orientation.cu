// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "magnetic_orientation.h"

#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/quaternion.h>

namespace mirheo
{

namespace magnetic_orientation_plugin_kernels
{
__global__ void applyMagneticField(ROVview view, real3 B, real3 M)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.nObjects) return;

    const auto q = static_cast<Quaternion<real>>(view.motions[gid].q);

    M = q.rotate(M);

    const real3 T = cross(M, B);

    atomicAdd(&view.motions[gid].torque.x, static_cast<RigidReal>(T.x));
    atomicAdd(&view.motions[gid].torque.y, static_cast<RigidReal>(T.y));
    atomicAdd(&view.motions[gid].torque.z, static_cast<RigidReal>(T.z));
}
} // namespace magnetic_orientation_plugin_kernels

MagneticOrientationPlugin::MagneticOrientationPlugin(const MirState *state, std::string name, std::string rovName,
                                                     real3 moment, UniformMagneticFunc magneticFunction) :
    SimulationPlugin(state, name),
    rovName_(rovName),
    moment_(moment),
    magneticFunction_(magneticFunction)
{}

void MagneticOrientationPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    rov_ = dynamic_cast<RigidObjectVector*>( simulation->getOVbyNameOrDie(rovName_) );
    if (rov_ == nullptr)
        die("Need rigid object vector to interact with magnetic field, plugin '%s', OV name '%s'",
            getCName(), rovName_.c_str());
}

void MagneticOrientationPlugin::beforeForces(cudaStream_t stream)
{
    ROVview view(rov_, rov_->local());
    const int nthreads = 128;

    const auto t = getState()->currentTime;
    const auto B = magneticFunction_(t);

    SAFE_KERNEL_LAUNCH(
            magnetic_orientation_plugin_kernels::applyMagneticField,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, B, moment_);
}

} // namespace mirheo
