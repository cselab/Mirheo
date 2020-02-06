#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

class ParticleDragPlugin : public SimulationPlugin
{
public:
    ParticleDragPlugin(const MirState *state, std::string name, std::string pvName, real drag);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_;
    real drag_;
};

} // namespace mirheo
