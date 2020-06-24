#pragma once

#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/folders.h>

namespace mirheo
{

class ParticleVector;

class TemperaturizePlugin : public SimulationPlugin
{
public:
    TemperaturizePlugin(const MirState *state, std::string name, std::string pvName, real kBT, bool keepVelocity);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_;
    real kBT_;
    bool keepVelocity_;
};

} // namespace mirheo
