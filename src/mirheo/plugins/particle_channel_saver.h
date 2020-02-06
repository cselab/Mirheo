#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

class ParticleChannelSaverPlugin : public SimulationPlugin
{
public:
    ParticleChannelSaverPlugin(const MirState *state, std::string name, std::string pvName,
                               std::string channelName, std::string savedName);

    void beforeIntegration(cudaStream_t stream) override;
    
    bool needPostproc() override;

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string pvName_;
    ParticleVector *pv_;
    std::string channelName_, savedName_;
};

} // namespace mirheo
