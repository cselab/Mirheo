// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class RigidObjectVector;

class AddTorquePlugin : public SimulationPlugin
{
public:
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
