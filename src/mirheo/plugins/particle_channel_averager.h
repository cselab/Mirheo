// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

/** Average over time a particle vector channel.
*/
class ParticleChannelAveragerPlugin : public SimulationPlugin
{
public:
    /** Create a ParticleChannelAveragerPlugin
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector.
        \param [in] channelName The name of the channel to average. Will fail if it does not exist.
        \param [in] averageName The name of the new channel, that will contain the time-averaged quantity..
        \param [in] updateEvery Will reset the averaged channel every this number of steps. Must be positive.
     */
    ParticleChannelAveragerPlugin(const MirState *state, std::string name, std::string pvName,
                                  std::string channelName, std::string averageName, real updateEvery);

    void beforeIntegration(cudaStream_t stream) override;

    bool needPostproc() override;

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string _makeWorkName() const;

private:
    std::string pvName_;
    ParticleVector *pv_ {nullptr};
    std::string channelName_; ///< Name of the original channel.
    std::string averageName_; ///< Name of the channel that will store the averaged value.
    std::string sumName_;     ///< Name of the work channel that stores the partial sum.

    int updateEvery_;
    int nSamples_{0};
};

} // namespace mirheo
