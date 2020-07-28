// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

/** Copies a given channel to another one that will "stick" to the particle vector.
    This is useful to collect statistics on non permanent quantities (e.g. stresses).
*/
class ParticleChannelSaverPlugin : public SimulationPlugin
{
public:
    /** Create a ParticleChannelSaverPlugin
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector.
        \param [in] channelName The name of the channel to save at every time step. Will fail if it does not exist.
        \param [in] savedName The name of the new channel.
     */
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
