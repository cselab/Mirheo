#include "add_force.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace AddForceKernels
{

__global__ void addForce(PVview view, float3 force)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    view.forces[gid] += make_float4(force, 0.0f);
}

} // namespace AddForceKernels

AddForcePlugin::AddForcePlugin(const MirState *state, std::string name, std::string pvName, float3 force) :
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

