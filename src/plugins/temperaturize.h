#pragma once

#include <plugins/interface.h>
#include <vector>
#include <string>

#include <core/utils/folders.h>

class ParticleVector;

class TemperaturizePlugin : public SimulationPlugin
{
public:
    TemperaturizePlugin(const MirState *state, std::string name, std::string pvName, float kbT, bool keepVelocity);
    
    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector* pv;
    float kbT;
    bool keepVelocity;
};

