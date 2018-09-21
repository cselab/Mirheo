#include "wall_repulsion.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/walls/simple_stationary_wall.h>
#include <core/simulation.h>

#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>


__global__ void forceFromSDF(PVview view, float* sdfs, float3* gradients, float C, float h, float maxForce)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p;
    p.readCoordinate(view.particles, pid);

    float sdf = sdfs[pid];
    float3 f = -gradients[pid] * min( maxForce, C * max(sdf + h, 0.0f) );

    view.forces[pid] += Float3_int(f, 0).toFloat4();
}


void WallRepulsionPlugin::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(sim, comm, interComm);

    pv = sim->getPVbyNameOrDie(pvName);
    wall = dynamic_cast<SDF_basedWall*>(sim->getWallByNameOrDie(wallName));

    if (wall == nullptr)
        die("Wall repulsion plugin '%s' can only work with SDF-based walls, but got wall '%s'",
                name.c_str(), wallName.c_str());
}


// TODO: make that force be computed on halo also
// to get rid of the SDF wall margin
void WallRepulsionPlugin::beforeIntegration(cudaStream_t stream)
{
    PVview view(pv, pv->local());

    wall->sdfPerParticle(pv->local(), &sdfs, &gradients, stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            forceFromSDF,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, sdfs.devPtr(), gradients.devPtr(), C, h, maxForce );
}

