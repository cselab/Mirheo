#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <string>

namespace mirheo
{

class RodVector;

class PinRodExtremityPlugin : public SimulationPlugin
{
public:
    PinRodExtremityPlugin(const MirState *state, std::string name, std::string rvName,
                          int segmentId, real fmagn, real3 targetDirection);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string rvName_;
    RodVector *rv_;
    int segmentId_;
    real fmagn_;
    real3 targetDirection_;
};

} // namespace mirheo
