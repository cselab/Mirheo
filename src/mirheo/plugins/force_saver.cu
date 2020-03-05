#include "force_saver.h"

#include <mirheo/core/simulation.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/pvs/views/pv.h>

namespace mirheo
{

const std::string ForceSaverPlugin::fieldName_ = "forces";

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
    SimulationPlugin(state, name),
    pvName_(pvName),
    pv_(nullptr)
{}

void ForceSaverPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    channel_names::failIfReserved(fieldName_, channel_names::reservedParticleFields);
    
    pv_->requireDataPerParticle<real3>(fieldName_, DataManager::PersistenceMode::None);
}

bool ForceSaverPlugin::needPostproc()
{
    return false;
}

void ForceSaverPlugin::beforeIntegration(cudaStream_t stream)
{
    auto savedForces  = pv_->local()->dataPerParticle.getData<real3>(fieldName_);
    PVview view(pv_, pv_->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            ForceSaverKernels::copyForces,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, savedForces->devPtr() );
}

} // namespace mirheo
