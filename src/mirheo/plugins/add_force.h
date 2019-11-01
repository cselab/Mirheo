#pragma once

#include "interface.h"

namespace mirheo
{

class ParticleVector;

class AddForcePlugin : public SimulationPlugin
{
public:
    AddForcePlugin(const MirState *state, std::string name, std::string pvName, real3 force);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector *pv;
    real3 force;
};

} // namespace mirheo
