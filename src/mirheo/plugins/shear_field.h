// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

#include <array>

namespace mirheo {

class ParticleVector;

/** Compute shear field at particles positions and store it in a channel.
*/
class ShearFieldPlugin : public SimulationPlugin
{
public:
    /** Create a ShearFieldPlugin
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector.
        \param [in] shear Shear tensor.
        \param [in] origin A point where the velocity is zero.
        \param [in] sfChannelName The name of the channel containing the shear field.
     */
    ShearFieldPlugin(const MirState *state, std::string name, std::string pvName,
                     std::array<real,9> shear, real3 origin, std::string sfChannelName);

    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override;

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string pvName_;
    ParticleVector *pv_;
    std::array<real,9> shear_;
    real3 origin_;
    std::string sfChannelName_;
};

} // namespace mirheo
