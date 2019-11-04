#include "add_force.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace AddForceKernels
{

__global__ void addForce(PVview view, real3 force)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    view.forces[gid] += make_real4(force, 0.0_r);
}

} // namespace AddForceKernels

AddForcePlugin::AddForcePlugin(const MirState *state, std::string name, std::string pvName, real3 force) :
    SimulationPlugin(state, name),
    pvName(pvName),
    force(force)
{}

void AddForcePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
}

void AddForcePlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            AddForceKernels::addForce,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, force );
}

} // namespace mirheo