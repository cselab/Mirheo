#pragma once

#include <string>
#include "interface.h"

class ParticleVector;

class ForceSaverPlugin : public SimulationPlugin
{
public:
    ForceSaverPlugin(const MirState *state, std::string name, std::string pvName);

    void beforeIntegration(cudaStream_t stream) override;
    
    bool needPostproc() override;

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string pvName;
    ParticleVector* pv;
    static const std::string fieldName;
};





