// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo {

class ParticleVector;

/** Add a constant force to every particle of a given ParticleVector at every time step.
    The force is added at the beforeForce() stage.
    The force has a reversed sign in half of the domain.
 */
class AddReversePoiseuilleForcePlugin : public SimulationPlugin
{
public:
    /** Create a AddReversePoiseuilleForcePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] force The force to apply.
        \param [in] flipDirection x, y or z. Indicates along which direction the force flips sign.
     */
    AddReversePoiseuilleForcePlugin(const MirState *state, const std::string& name,
                                    const std::string& pvName, real3 force, char flipDirection);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_ {nullptr};
    real3 force_;
    char flipDirection_;
};

} // namespace mirheo
