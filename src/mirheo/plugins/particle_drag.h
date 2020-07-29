// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

/** Apply a drag force proportional to the velocity of every particle in a ParticleVector.
 */
class ParticleDragPlugin : public SimulationPlugin
{
public:
    /** Create a ParticleDragPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] drag The drag coefficient applied to each particle.
     */
    ParticleDragPlugin(const MirState *state, std::string name, std::string pvName, real drag);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_;
    real drag_;
};

} // namespace mirheo
