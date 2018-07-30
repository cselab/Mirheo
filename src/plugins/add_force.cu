#include "add_force.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>

#include <core/utils/cuda_common.h>

__global__ void addForce(PVview view, float3 force)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    view.forces[gid] += make_float4(force, 0.0f);
}

void AddForcePlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(sim, comm, interComm);

    pv = sim->getPVbyNameOrDie(pvName);
}

void AddForcePlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            addForce,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, force );
}

