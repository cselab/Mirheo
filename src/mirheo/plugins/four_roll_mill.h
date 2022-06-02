// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

/** Add the "four roll mill" force to every particle of a given ParticleVector at every time step.
    The force is added at the beforeForce() stage.
    The force has the form f = A * (sin(x) cos(y), -cos(x) sin(y), 0)
    where A is the intensity of the force and x, y are scaled so that they cover one period over the domain.
 */
class AddFourRollMillForcePlugin : public SimulationPlugin
{
public:
    /** Create a AddFourRollMillForcePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] intensity The intensity A of the force to apply.
     */
    AddFourRollMillForcePlugin(const MirState *state, const std::string& name, const std::string& pvName, real intensity);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_ {nullptr};
    real intensity_;
};

} // namespace mirheo
