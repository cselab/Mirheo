#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

#include <functional>

class ParticleVector;

using FuncTime3D = std::function<float3(float)>;

class AnchorParticlePlugin : public SimulationPlugin
{
public:
    AnchorParticlePlugin(const YmrState *state, std::string name, std::string pvName,
                         FuncTime3D position, FuncTime3D velocity, int pid);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector *pv;

    FuncTime3D position;
    FuncTime3D velocity;
    int pid;
};

