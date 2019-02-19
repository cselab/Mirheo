#pragma once

#include "interface.h"

#include <string>

class ParticleVector;

class ParticleDisplacement : public SimulationPlugin
{
public:
    ParticleDisplacement(const YmrState *state, std::string name, std::string pvName, int updateEvery);
    ~ParticleDisplacement();

    void beforeIntegration(cudaStream_t stream) override;
    
    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override {return false;}
    
private:

    std::string pvName;
    ParticleVector *pv;
    int updateEvery;

    const std::string displacementChannelName = "displacement";
    const std::string savedPositionChannelName;
};
