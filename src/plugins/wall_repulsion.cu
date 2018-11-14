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

    if (sdf + h >= 0.0f)
    {
        float3 f = -gradients[pid] * min( maxForce, C * max(sdf + h, 0.0f) );
        atomicAdd(view.forces + pid, f);
    }
}


void WallRepulsionPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
    wall = dynamic_cast<SDF_basedWall*>(simulation->getWallByNameOrDie(wallName));
    
    pv->requireDataPerParticle<float>("sdf", false);
    pv->requireDataPerParticle<float3>("grad_sdf", false);

    if (wall == nullptr)
        die("Wall repulsion plugin '%s' can only work with SDF-based walls, but got wall '%s'",
            name.c_str(), wallName.c_str());
}


// TODO: make that force be computed on halo also
// to get rid of the SDF wall margin
void WallRepulsionPlugin::beforeIntegration(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    
    auto sdfs      = pv->local()->extraPerParticle.getData<float>("sdf");
    auto gradients = pv->local()->extraPerParticle.getData<float3>("grad_sdf");

    wall->sdfPerParticle(pv->local(), sdfs, gradients, h+0.1f, stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            forceFromSDF,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, sdfs->devPtr(), gradients->devPtr(), C, h, maxForce );
}

