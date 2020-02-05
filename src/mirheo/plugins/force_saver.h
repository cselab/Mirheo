#pragma once

#include <mirheo/core/plugins.h>
#include <string>

namespace mirheo
{

class ParticleVector;

class ForceSaverPlugin : public SimulationPlugin
{
public:
    ForceSaverPlugin(const MirState *state, std::string name, std::string pvName);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    bool needPostproc() override;
    void beforeIntegration(cudaStream_t stream) override;

private:
    std::string pvName_;
    ParticleVector *pv_;
    static const std::string fieldName_;
};

} // namespace mirheo
