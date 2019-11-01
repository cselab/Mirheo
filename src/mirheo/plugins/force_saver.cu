#include "force_saver.h"

#include <mirheo/core/simulation.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/pvs/views/pv.h>

namespace mirheo
{

const std::string ForceSaverPlugin::fieldName = "forces";

namespace ForceSaverKernels
{

__global__ void copyForces(PVview view, real3 *savedForces)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    real3 f = Real3_int(view.forces[pid]).v;
    savedForces[pid] = f;
}

} // namespace ForceSaverKernels

ForceSaverPlugin::ForceSaverPlugin(const MirState *state, std::string name, std::string pvName) :
    SimulationPlugin(state, name), pvName(pvName), pv(nullptr)
{}

void ForceSaverPlugin::beforeIntegration(cudaStream_t stream)
{
    auto savedForces  = pv->local()->dataPerParticle.getData<real3>(fieldName);
    PVview view(pv, pv->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            ForceSaverKernels::copyForces,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, savedForces->devPtr() );
}
    
bool ForceSaverPlugin::needPostproc()
{
    return false;
}

void ForceSaverPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    pv->requireDataPerParticle<real3>(fieldName, DataManager::PersistenceMode::None);
}

} // namespace mirheo
