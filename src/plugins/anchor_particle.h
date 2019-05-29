#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/utils/pytypes.h>

#include <functional>

class ParticleVector;

using FuncTime3D = std::function<float3(float)>;

class AnchorParticlesPlugin : public SimulationPlugin
{
public:
    AnchorParticlesPlugin(const YmrState *state, std::string name, std::string pvName,
                          FuncTime3D position, FuncTime3D velocity, int pid, int reportEvery);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName;
    ParticleVector *pv;

    FuncTime3D position;
    FuncTime3D velocity;
    int pid;

    PinnedBuffer<double3> force {1};
    int nsamples {0};
    int reportEvery;
    std::vector<char> sendBuffer;
};




class AnchorParticlesStatsPlugin : public PostprocessPlugin
{
public:
    AnchorParticlesStatsPlugin(std::string name, std::string path);
    ~AnchorParticlesStatsPlugin();
    
    void deserialize(MPI_Status& stat) override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    bool activated;
    std::string path;

    FILE *fout {nullptr};
    float3 force;
};
