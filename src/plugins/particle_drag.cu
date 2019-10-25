#include "particle_drag.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace ParticleDragPluginKernels
{

__global__ void applyDrag(PVview view, real drag)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    auto v = make_real3(view.readVelocity(gid));
    auto force = - drag * v;
    view.forces[gid] += make_real4(force, 0.0f);
}

} // namespace ParticleDragPluginKernels

ParticleDragPlugin::ParticleDragPlugin(const MirState *state, std::string name, std::string pvName, real drag) :
    SimulationPlugin(state, name),
    pvName(pvName),
    drag(drag)
{}

void ParticleDragPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
}

void ParticleDragPlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            ParticleDragPluginKernels::applyDrag,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, drag );
}

