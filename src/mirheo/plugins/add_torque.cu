#include "add_torque.h"

#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace AddTorqueKernels
{

__global__ void addTorque(ROVview view, real3 torque)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.nObjects) return;

    view.motions[gid].torque += torque;
}

} // namespace AddTorqueKernels

AddTorquePlugin::AddTorquePlugin(const MirState *state, const std::string& name, const std::string& rovName, real3 torque) :
    SimulationPlugin(state, name),
    rovName_(rovName),
    torque_(torque)
{}

void AddTorquePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    rov_ = dynamic_cast<RigidObjectVector*>( simulation->getOVbyNameOrDie(rovName_) );
    if (rov_ == nullptr)
        die("Need rigid object vector to add torque, plugin '%s', OV name '%s'",
            getCName(), rovName_.c_str());

    info("Objects '%s' will experience external torque [%f %f %f]", 
            rovName_.c_str(), torque_.x, torque_.y, torque_.z);
}

void AddTorquePlugin::beforeForces(cudaStream_t stream)
{
    ROVview view(rov_, rov_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            AddTorqueKernels::addTorque,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, torque_ );
}

} // namespace mirheo
