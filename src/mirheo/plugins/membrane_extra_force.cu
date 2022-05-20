// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "membrane_extra_force.h"

#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <fstream>

namespace mirheo
{

namespace membrane_extra_forces_kernels
{
__global__ void addForce(OVview view, const Force *forces)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    int locId = gid % view.objSize;

    view.forces[gid] += forces[locId].toReal4();
}
} // namespace membrane_extra_forces_kernels

MembraneExtraForcePlugin::MembraneExtraForcePlugin(const MirState *state, std::string name,
                                                   std::string pvName, const std::vector<real3>& forces) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    forces_(forces.size())
{
    HostBuffer<Force> hostForces(forces.size());

    for (size_t i = 0; i < forces.size(); ++i)
        hostForces[i].f = forces[i];

    forces_.copy(hostForces, defaultStream);
}

void MembraneExtraForcePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    auto pvPtr = simulation->getPVbyNameOrDie(pvName_);
    if ( !(pv_ = dynamic_cast<MembraneVector*>(pvPtr)) )
        die("MembraneExtraForcePlugin '%s' expects a MembraneVector (given '%s')",
            getCName(), pvName_.c_str());
}

void MembraneExtraForcePlugin::beforeForces(cudaStream_t stream)
{
    OVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        membrane_extra_forces_kernels::addForce,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, forces_.devPtr() );
}


} // namespace mirheo
