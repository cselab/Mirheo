#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <vector>
#include <string>

#include <core/utils/folders.h>

class ParticleVector;

class ImposeVelocityPlugin : public SimulationPlugin
{
public:
    ImposeVelocityPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                         real3 low, real3 high, real3 targetVel, int every);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }
    
    void setTargetVelocity(real3 v);
    
private:
    std::vector<std::string> pvNames;
    std::vector<ParticleVector*> pvs;

    real3 high, low;
    real3 targetVel;

    int every;

    PinnedBuffer<int> nSamples{1};
    PinnedBuffer<double3> totVel{1};
};

