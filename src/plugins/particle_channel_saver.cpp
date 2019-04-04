#include "particle_channel_saver.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/type_map.h>

ParticleChannelSaverPlugin::ParticleChannelSaverPlugin(const YmrState *state, std::string name, std::string pvName,
                                                       std::string channelName, std::string savedName) :
    SimulationPlugin(state, name),
    pvName(pvName),
    pv(nullptr),
    channelName(channelName),
    savedName(savedName)
{}

void ParticleChannelSaverPlugin::beforeIntegration(cudaStream_t stream)
{
    auto& extraManager = pv->local()->extraPerParticle;
    const auto& srcDesc = extraManager.getChannelDescOrDie(channelName);
    const auto& dstDesc = extraManager.getChannelDescOrDie(savedName);

    mpark::visit([&](auto srcBufferPtr) {
                     auto dstBufferPtr = mpark::get<decltype(srcBufferPtr)>(dstDesc.varDataPtr);
                     dstBufferPtr->copyDeviceOnly(*srcBufferPtr, stream);
                 }, srcDesc.varDataPtr);
}
    
bool ParticleChannelSaverPlugin::needPostproc()
{
    return false;
}

void ParticleChannelSaverPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);

    const auto& desc = pv->local()->extraPerParticle.getChannelDescOrDie(channelName);

    mpark::visit([&](auto pinnedBufferPtr) {
                     using T = typename std::remove_reference< decltype(*pinnedBufferPtr->hostPtr()) >::type;
                     pv->requireDataPerParticle<T>(savedName,ExtraDataManager::PersistenceMode::Persistent);
                 }, desc.varDataPtr);
}
