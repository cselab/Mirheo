// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo {

class ParticleVector;

/** Compute the exponential moving average (EMA) of the particles velocity.
*/
class ExponentialMovingAveragePlugin : public SimulationPlugin
{
public:
    /** Create a ExponentialMovingAveragePlugin
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector.
        \param [in] alpha Coefficient in [0,1] to perform the EMA.
        \param [in] srcChannelName The name of the channel to average. Will fail if it does not exist.
        \param [in] emaChannelName The name of the channel containing the EMA.
     */
    ExponentialMovingAveragePlugin(const MirState *state, std::string name, std::string pvName,
                                   real alpha, std::string srcChannelName, std::string emaChannelName);

    void beforeIntegration(cudaStream_t stream) override;

    bool needPostproc() override;

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string pvName_;
    ParticleVector *pv_;
    real alpha_;
    std::string srcChannelName_;
    std::string emaChannelName_;
};

} // namespace mirheo
