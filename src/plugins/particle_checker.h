#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>

#include <vector>

class ParticleVector;

class ParticleCheckerPlugin : public SimulationPlugin
{
public:
    ParticleCheckerPlugin(const YmrState *state, std::string name, int checkEvery);
    ~ParticleCheckerPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

    enum class Status {Ok, Out, Nan};
    
    struct __align__(8) ParticleStatus
    {
        Status status;
        int id;
    };

private:
    int checkEvery;
    
    PinnedBuffer<ParticleStatus> statuses;
    std::vector<ParticleVector*> pvs;
};
