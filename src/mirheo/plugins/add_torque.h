// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class RigidObjectVector;

/** Add a constant torque to every object of a given RigidObjectVector at every time step.
    The torque is added at the beforeForce() stage.
 */
class AddTorquePlugin : public SimulationPlugin
{
public:
    /** Create a AddTorquePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] rovName The name of the RigidObjectVector to which the torque should be applied.
        \param [in] torque The torque to apply.
     */
    AddTorquePlugin(const MirState *state, const std::string& name, const std::string& rovName, real3 torque);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string rovName_;
    RigidObjectVector *rov_ {nullptr};
    real3 torque_;
};

} // namespace mirheo
