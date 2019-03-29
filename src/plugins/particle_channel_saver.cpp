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
    const auto& desc = extraManager.getChannelDescOrDie(channelName);    
    
#define SWITCH_ENTRY(ctype) case DataType::TOKENIZE(ctype):             \
    {                                                                   \
        auto src = extraManager.getData<ctype>(channelName);            \
        auto dst = extraManager.getData<ctype>(savedName);              \
        dst->copyDeviceOnly(*src, stream);                              \
        break;                                                          \
    }

    switch(desc.dataType) {
        TYPE_TABLE(SWITCH_ENTRY);
    }

#undef SWITCH_ENTRY
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

#define SWITCH_ENTRY(ctype) case DataType::TOKENIZE(ctype):             \
    pv->requireDataPerParticle<ctype>(savedName,                        \
                                      ExtraDataManager::PersistenceMode::Persistent); \
    break;
    
    switch(desc.dataType)
    {
        TYPE_TABLE(SWITCH_ENTRY);
    default:
        die("cannot save field '%s' from pv '%s': unknown type",
            channelName.c_str(), pvName.c_str());
        break;
    }

#undef SWITCH_ENTRY
}

    
