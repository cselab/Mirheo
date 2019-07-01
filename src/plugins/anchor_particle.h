#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/utils/pytypes.h>

#include <functional>
#include <vector>

class ParticleVector;

using FuncTime3D = std::function<std::vector<float3>(float)>;

class AnchorParticlesPlugin : public SimulationPlugin
{
public:
    AnchorParticlesPlugin(const MirState *state, std::string name, std::string pvName,
                          FuncTime3D positions, FuncTime3D velocities,
                          std::vector<int> pids, int reportEvery);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName;
    ParticleVector *pv;

    FuncTime3D positions;
    FuncTime3D velocities;

    PinnedBuffer<double3> forces;
    PinnedBuffer<float3> posBuffer, velBuffer;
    PinnedBuffer<int> pids;

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
};
