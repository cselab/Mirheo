#pragma once

#include "interface.h"

#include <string>

class ParticleVector;

class ParticleDisplacementPlugin : public SimulationPlugin
{
public:
    ParticleDisplacementPlugin(const YmrState *state, std::string name, std::string pvName, int updateEvery);
    ~ParticleDisplacementPlugin();

    void afterIntegration(cudaStream_t stream) override;
    
    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override {return false;}
    
private:

    std::string pvName;
    ParticleVector *pv;
    int updateEvery;

    const std::string displacementChannelName = "displacements";
    const std::string savedPositionChannelName;
};
