// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

#include <functional>
#include <memory>

namespace mirheo {

class ParticleVector;
class VectorField;

/** Add a force to every particle of a given ParticleVector at every time step.
    The force is a provided vector field evaluated at the particle's position.
    The force is added at the beforeForce() stage.
 */
class AddForceFieldPlugin : public SimulationPlugin
{
public:
    /// functor that describes a force field on the CPU.
    using ForceField = std::function<real3(real3)>;

    /** Create a AddForceFieldPlugin object from a functor.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] forceField The force field.
        \param [in] gridSpacing The grid spacing used to discretize \p potentialField.
     */
    AddForceFieldPlugin(const MirState *state, const std::string& name,
                        const std::string& pvName,
                        ForceField forceField,
                        real3 gridSpacing);

    /** Create a AddForceFieldPlugin object from a file.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] forceFieldFilename The file containing the force field on a cartesian grid. See VectorFieldFromFile.
        \param [in] gridSpacing The grid spacing used to discretize \p potentialField.
     */
    AddForceFieldPlugin(const MirState *state,
                        const std::string& name,
                        const std::string& pvName,
                        std::string forceFieldFilename,
                        real3 gridSpacing);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_ {nullptr};
    std::unique_ptr<VectorField> forceField_; /// the force field
};

} // namespace mirheo
