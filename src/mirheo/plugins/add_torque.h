#pragma once

#include "interface.h"

namespace mirheo
{

class RigidObjectVector;

class AddTorquePlugin : public SimulationPlugin
{
public:
    AddTorquePlugin(const MirState *state, std::string name, std::string rovName, real3 torque);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string rovName;
    RigidObjectVector *rov;
    real3 torque;
};

} // namespace mirheo
