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

AddTorquePlugin::AddTorquePlugin(const MirState *state, std::string name, std::string rovName, real3 torque) :
    SimulationPlugin(state, name),
    rovName(rovName),
    torque(torque)
{}

void AddTorquePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    rov = dynamic_cast<RigidObjectVector*>( simulation->getOVbyNameOrDie(rovName) );
    if (rov == nullptr)
        die("Need rigid object vector to add torque, plugin '%s', OV name '%s'",
            name.c_str(), rovName.c_str());

    info("Objects '%s' will experience external torque [%f %f %f]", 
            rovName.c_str(), torque.x, torque.y, torque.z);
}

void AddTorquePlugin::beforeForces(cudaStream_t stream)
{
    ROVview view(rov, rov->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            AddTorqueKernels::addTorque,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, torque );
}

} // namespace mirheo
