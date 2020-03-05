#include "particle_channel_saver.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

ParticleChannelSaverPlugin::ParticleChannelSaverPlugin(const MirState *state, std::string name, std::string pvName,
                                                       std::string channelName, std::string savedName) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    pv_(nullptr),
    channelName_(channelName),
    savedName_(savedName)
{
    channel_names::failIfReserved(savedName_, channel_names::reservedParticleFields);
}

void ParticleChannelSaverPlugin::beforeIntegration(cudaStream_t stream)
{
    auto& dataManager = pv_->local()->dataPerParticle;
    const auto& srcDesc = dataManager.getChannelDescOrDie(channelName_);
    const auto& dstDesc = dataManager.getChannelDescOrDie(savedName_);

    mpark::visit([&](auto srcBufferPtr)
    {
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

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    const auto& desc = pv_->local()->dataPerParticle.getChannelDescOrDie(channelName_);

    mpark::visit([&](auto pinnedBufferPtr)
    {
        using T = typename std::remove_reference< decltype(*pinnedBufferPtr->hostPtr()) >::type;
        pv_->requireDataPerParticle<T>(savedName_, DataManager::PersistenceMode::Active);
    }, desc.varDataPtr);
}

} // namespace mirheo
