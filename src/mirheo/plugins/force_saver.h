// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>
#include <string>

namespace mirheo
{

class ParticleVector;

/** Copies the forces of a given ParticleVector to a new channel at every time step.
    This allows to dump the forces since they are reset to zero at every time step.
*/
class ForceSaverPlugin : public SimulationPlugin
{
public:
    /** Create a ForceSaverPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to save forces from and to.
     */
    ForceSaverPlugin(const MirState *state, std::string name, std::string pvName);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    bool needPostproc() override;
    void beforeIntegration(cudaStream_t stream) override;

private:
    std::string pvName_;
    ParticleVector *pv_;
    static const std::string fieldName_;
};

} // namespace mirheo
