#include "displacement.h"
#include "utils/time_stamp.h"

#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/kernel_launch.h>

namespace ParticleDisplacementPluginKernels
{

__global__ void extractPositions(PVview view, real4 *positions)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > view.size) return;
    positions[i] = view.readPosition(i);
}

__global__ void computeDisplacementsAndSavePositions(PVview view, real4 *positions, real3 *displacements)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > view.size) return;
    
    Real3_int pos(view.readPosition(i));
    Real3_int oldPos(positions[i]);
    
    displacements[i] = pos.v - oldPos.v;
    positions[i] = pos.toReal4();
}

} // namespace DisplacementKernels

ParticleDisplacementPlugin::ParticleDisplacementPlugin(const MirState *state, std::string name, std::string pvName, int updateEvery) :
    SimulationPlugin(state, name),
    pvName(pvName),
    pv(nullptr),
    updateEvery(updateEvery)
{}

ParticleDisplacementPlugin::~ParticleDisplacementPlugin() = default;

void ParticleDisplacementPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    pv->requireDataPerParticle<real3>(displacementChannelName,
                                       DataManager::PersistenceMode::Active);

    pv->requireDataPerParticle<real4>(savedPositionChannelName,
                                       DataManager::PersistenceMode::Active,
                                       DataManager::ShiftMode::Active);

    PVview view(pv, pv->local());
    const int nthreads = 128;    

    auto& manager      = pv->local()->dataPerParticle;
    auto positions     = manager.getData<real4>(savedPositionChannelName);
    auto displacements = manager.getData<real3>(displacementChannelName);

    displacements->clear(defaultStream);
    
    SAFE_KERNEL_LAUNCH(
            ParticleDisplacementPluginKernels::extractPositions,
            getNblocks(view.size, nthreads), nthreads, 0, defaultStream,
            view, positions->devPtr());
}

void ParticleDisplacementPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(state, updateEvery)) return;

    auto& manager = pv->local()->dataPerParticle;
    
    auto positions     = manager.getData<real4>(savedPositionChannelName);
    auto displacements = manager.getData<real3>( displacementChannelName);

    PVview view(pv, pv->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            ParticleDisplacementPluginKernels::computeDisplacementsAndSavePositions,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, positions->devPtr(), displacements->devPtr());    
}
