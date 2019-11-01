#pragma once

#include <mirheo/plugins/interface.h>
#include <vector>
#include <string>

#include <mirheo/core/utils/folders.h>

namespace mirheo
{

class ParticleVector;

class TemperaturizePlugin : public SimulationPlugin
{
public:
    TemperaturizePlugin(const MirState *state, std::string name, std::string pvName, real kBT, bool keepVelocity);
    
    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector* pv;
    real kBT;
    bool keepVelocity;
};

} // namespace mirheo
