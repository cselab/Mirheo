// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

#include <functional>
#include <memory>

namespace mirheo {

class ParticleVector;
class ScalarField;

/** Add a force to every particle of a given ParticleVector at every time step.
    The force is the derivative of a provided scalar field at the particle's position.
    The force is added at the beforeForce() stage.
 */
class AddPressureGradientPlugin : public SimulationPlugin
{
public:
    /// functor that describes a pressure field on the CPU.
    using PressureField = std::function<real(real3)>;

    /** Create a AddPressureGradientPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to which the force should be applied.
        \param [in] pressureField The pressure scalar field.
        \param [in] gridSpacing The grid spacing used to discretize \p pressureField.
     */
    AddPressureGradientPlugin(const MirState *state, const std::string& name,
                              const std::string& pvName,
                              PressureField pressureField,
                              real3 gridSpacing);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_ {nullptr};
    std::unique_ptr<ScalarField> pressureField_; /// the pressure field
};

} // namespace mirheo
