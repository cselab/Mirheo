// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo {

class ParticleVector;

/** Compute sinusoidal field at particles positions and store it in a channel.
*/
class SinusoidalFieldPlugin : public SimulationPlugin
{
public:
    /** Create a SinusoidalFieldPlugin
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector.
        \param [in] magnitude Maximum velocity along x.
        \param [in] waveNumber Number of periods along y.
        \param [in] sfChannelName The name of the channel containing the shear field.
     */
    SinusoidalFieldPlugin(const MirState *state, std::string name, std::string pvName,
                          real magnitude, int waveNumber, std::string sfChannelName);

    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override;

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string pvName_;
    ParticleVector *pv_;
    real magnitude_;
    int waveNumber_;
    std::string sfChannelName_;
};

} // namespace mirheo
