#pragma once

#include "interface.h"
#include "utils/pid.h"

#include <core/containers.h>
#include <core/datatypes.h>

#include <vector>

class ParticleVector;

class SimulationVelocityControl : public SimulationPlugin
{
public:
    SimulationVelocityControl(const YmrState *state, std::string name, std::vector<std::string> pvNames,
                              float3 low, float3 high,
                              int sampleEvery, int tuneEvery, int dumpEvery,
                              float3 targetVel, float Kp, float Ki, float Kd);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

    void checkpoint(MPI_Comm comm, std::string path, CheckpointIdAdvanceMode checkpointMode) override;
    void restart(MPI_Comm comm, std::string path) override;

    
private:
    int sampleEvery, dumpEvery, tuneEvery;
    std::vector<std::string> pvNames;
    std::vector<ParticleVector*> pvs;

    float3 high, low;
    float3 currentVel, targetVel, force;

    PinnedBuffer<int> nSamples{1};
    PinnedBuffer<float3> totVel{1};
    double3 accumulatedTotVel;
    

    PidControl<float3> pid;
    std::vector<char> sendBuffer;

private:
    void sampleOnePv(ParticleVector *pv, cudaStream_t stream);
};

class PostprocessVelocityControl : public PostprocessPlugin
{
public:
    PostprocessVelocityControl(std::string name, std::string filename);
    ~PostprocessVelocityControl();

    void deserialize(MPI_Status& stat) override;

private:
    FILE *fdump;
};
