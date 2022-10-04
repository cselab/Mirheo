// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo {

class ParticleVector;

/** Add a sinusoidal (Kolmogorov) force to every particle of a given ParticleVector at every time step.
    The force is added at the beforeForce() stage.
    The force has a reversed sign in half of the domain.
 */
class AddSinusoidalForcePlugin : public SimulationPlugin
{
public:
    /** Create a AddSinusoidalForcePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] magnitude Magnitude of the force.
        \param [in] waveNumber How many wavelengths along y.
     */
    AddSinusoidalForcePlugin(const MirState *state, const std::string& name,
                             const std::string& pvName, real magnitude, int waveNumber);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_ {nullptr};
    real magnitude_;
    int waveNumber_;
};

} // namespace mirheo
