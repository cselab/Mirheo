#include "displacement.h"
#include "utils/time_stamp.h"

#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/kernel_launch.h>

namespace ParticleDisplacementPluginKernels
{

__global__ void extractPositions(PVview view, float4 *positions)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > view.size) return;
    positions[i] = view.readPosition(i);
}

__global__ void computeDisplacementsAndSavePositions(PVview view, float4 *positions, float3 *displacements)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > view.size) return;
    
    Float3_int pos(view.readPosition(i));
    Float3_int oldPos(positions[i]);
    
    displacements[i] = pos.v - oldPos.v;
    positions[i] = pos.toFloat4();
}

} // namespace DisplacementKernels

ParticleDisplacementPlugin::ParticleDisplacementPlugin(const YmrState *state, std::string name, std::string pvName, int updateEvery) :
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

    pv->requireDataPerParticle<float3>(displacementChannelName,
                                       ExtraDataManager::PersistenceMode::Persistent);

    pv->requireDataPerParticle<float4>(savedPositionChannelName,
                                       ExtraDataManager::PersistenceMode::Persistent,
                                       sizeof(float4::x));

    PVview view(pv, pv->local());
    const int nthreads = 128;    

    auto& manager      = pv->local()->extraPerParticle;
    auto positions     = manager.getData<float4>(savedPositionChannelName);
    auto displacements = manager.getData<float3>(displacementChannelName);

    displacements->clear(defaultStream);
    
    SAFE_KERNEL_LAUNCH(
            ParticleDisplacementPluginKernels::extractPositions,
            getNblocks(view.size, nthreads), nthreads, 0, defaultStream,
            view, positions->devPtr());
}

void ParticleDisplacementPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(state, updateEvery)) return;

    auto& manager = pv->local()->extraPerParticle;
    
    auto positions     = manager.getData<float4>(savedPositionChannelName);
    auto displacements = manager.getData<float3>( displacementChannelName);

    PVview view(pv, pv->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            ParticleDisplacementPluginKernels::computeDisplacementsAndSavePositions,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, positions->devPtr(), displacements->devPtr());    
}
