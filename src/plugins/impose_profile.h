#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <vector>
#include <string>

#include <core/utils/folders.h>

class ParticleVector;
class CellList;

class ImposeProfilePlugin : public SimulationPlugin
{
public:
    ImposeProfilePlugin(std::string name, const YmrState *state, std::string pvName, float3 low, float3 high, float3 targetVel, float kbT) :
        SimulationPlugin(name, state), pvName(pvName), low(low), high(high), targetVel(targetVel), kbT(kbT)
    {}

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector* pv;
    CellList* cl;

    float3 high, low;
    float3 targetVel;
    float kbT;

    PinnedBuffer<int> nRelevantCells{1};
    DeviceBuffer<int> relevantCells;
};

