#include "anchor_particle.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace AnchorParticleKernels
{

__global__ void anchorParticle(PVview view, int pid, float3 pos, float3 vel)
{
    if (threadIdx.x > 0) return;

    Particle p = view.readParticle(pid);
    p.r = pos;
    p.u = vel;
    view.writeParticle(pid, p);
}

} // namespace AnchorParticleKernels

AnchorParticlePlugin::AnchorParticlePlugin(const YmrState *state, std::string name, std::string pvName,
                                           FuncTime3D position, FuncTime3D velocity, int pid) :
    SimulationPlugin(state, name),
    pvName(pvName),
    position(position),
    velocity(velocity),
    pid(pid)
{
    if (pid < 0)
        die("invalid particle id %d\n", pid);
}

void AnchorParticlePlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
}

void AnchorParticlePlugin::afterIntegration(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    const int nthreads = 32;
    const int nblocks = 1;

    if (view.size == 0) return;
    if (pid >= view.size) return;

    float t = (float) state->currentTime;
    const auto& domain = state->domain;

    float3 pos = position(t);
    float3 vel = velocity(t);
    pos = domain.global2local(pos);
    
    SAFE_KERNEL_LAUNCH(
            AnchorParticleKernels::anchorParticle,
            nblocks, nthreads, 0, stream,
            view, pid, pos, vel );
}


