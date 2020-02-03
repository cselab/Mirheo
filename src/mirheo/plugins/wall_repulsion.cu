#include "wall_repulsion.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/walls/simple_stationary_wall.h>

namespace mirheo
{

namespace ChannelNames
{
static const std::string      sdf =      "sdf";
static const std::string grad_sdf = "grad_sdf";
} // namespace ChannelNames

namespace WallRepulsionPluginKernels
{
__global__ void forceFromSDF(PVview view, const real *sdfs, const real3 *gradients, real C, real h, real maxForce)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    const real sdf = sdfs[pid];

    if (sdf + h >= 0.0_r)
    {
        const real3 f = -gradients[pid] * math::min( maxForce, C * math::max(sdf + h, 0.0_r) );
        atomicAdd(view.forces + pid, f);
    }
}
} // WallRepulsionPluginKernels

WallRepulsionPlugin::WallRepulsionPlugin(const MirState *state, std::string name,
                                         std::string pvName, std::string wallName,
                                         real C, real h, real maxForce) :
    SimulationPlugin(state, name),
    pvName(pvName),
    wallName(wallName),
    C(C),
    h(h),
    maxForce(maxForce)
{}

void WallRepulsionPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
    wall = dynamic_cast<SDF_basedWall*>(simulation->getWallByNameOrDie(wallName));
    
    pv->requireDataPerParticle<real>(ChannelNames::sdf, DataManager::PersistenceMode::None);
    pv->requireDataPerParticle<real3>(ChannelNames::grad_sdf, DataManager::PersistenceMode::None);

    if (wall == nullptr)
        die("Wall repulsion plugin '%s' can only work with SDF-based walls, but got wall '%s'",
            getCName(), wallName.c_str());
}


// TODO: make that force be computed on halo also
// to get rid of the SDF wall margin
void WallRepulsionPlugin::beforeIntegration(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    
    auto sdfs      = pv->local()->dataPerParticle.getData<real>(ChannelNames::sdf);
    auto gradients = pv->local()->dataPerParticle.getData<real3>(ChannelNames::grad_sdf);

    const real gradientThreshold = h + 0.1_r;
    
    wall->sdfPerParticle(pv->local(), sdfs, gradients, gradientThreshold, stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            WallRepulsionPluginKernels::forceFromSDF,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, sdfs->devPtr(), gradients->devPtr(), C, h, maxForce );
}

} // namespace mirheo
