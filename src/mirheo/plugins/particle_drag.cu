#include "particle_drag.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace ParticleDragPluginKernels
{

__global__ void applyDrag(PVview view, real drag)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    auto v = make_real3(view.readVelocity(gid));
    auto force = - drag * v;
    view.forces[gid] += make_real4(force, 0.0_r);
}

} // namespace ParticleDragPluginKernels

ParticleDragPlugin::ParticleDragPlugin(const MirState *state, std::string name, std::string pvName, real drag) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    drag_(drag)
{}

void ParticleDragPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);
}

void ParticleDragPlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            ParticleDragPluginKernels::applyDrag,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, drag_ );
}

} // namespace mirheo
