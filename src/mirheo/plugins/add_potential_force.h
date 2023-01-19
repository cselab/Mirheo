// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

#include <functional>
#include <memory>

namespace mirheo {

class ParticleVector;
class ScalarField;

/** Add a force to every particle of a given ParticleVector at every time step.
    The force is the negative gradient of a provided scalar field at the particle's position.
    The force is added at the beforeForce() stage.
 */
class AddPotentialForcePlugin : public SimulationPlugin
{
public:
    /// functor that describes a pressure field on the CPU.
    using PotentialField = std::function<real(real3)>;

    /** Create a AddPotentialForcePlugin object from a functor.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] potentialField The potential scalar field.
        \param [in] gridSpacing The grid spacing used to discretize \p potentialField.
     */
    AddPotentialForcePlugin(const MirState *state, const std::string& name,
                            const std::string& pvName,
                            PotentialField potentialField,
                            real3 gridSpacing);

    /** Create a AddPotentialForcePlugin object from a file.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] potentialFieldFilename The file containing the potential scalar field on a cartesian grid. See ScalarFieldFromFile.
        \param [in] gridSpacing The grid spacing used to discretize \p potentialField.
     */
    AddPotentialForcePlugin(const MirState *state,
                            const std::string& name,
                            const std::string& pvName,
                            std::string potentialFieldFilename,
                            real3 gridSpacing);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_ {nullptr};
    std::unique_ptr<ScalarField> potentialField_; /// the potential field
};

} // namespace mirheo
