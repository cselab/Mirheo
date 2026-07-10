// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

/** Add a per-particle force, read from a channel of the ParticleVector,
    to every particle at every time step.
    The force is added at the beforeForce() stage.
 */
class AddPerParticleForcePlugin : public SimulationPlugin
{
public:
    /** Create an AddPerParticleForcePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the forces should be applied.
        \param [in] channel_name The name of the channel that contains the per-particle forces.
     */
    AddPerParticleForcePlugin(const MirState *state, const std::string& name, const std::string& pvName, const std::string& channel_name);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_ {nullptr};
    std::string channel_name_;
};

} // namespace mirheo
