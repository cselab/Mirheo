#pragma once

#include <mirheo/plugins/interface.h>
#include <mirheo/core/containers.h>
#include <vector>
#include <string>

#include <mirheo/core/utils/folders.h>

class ParticleVector;
class CellList;

class ImposeProfilePlugin : public SimulationPlugin
{
public:
    ImposeProfilePlugin(const MirState *state, std::string name, std::string pvName,
                        real3 low, real3 high, real3 targetVel, real kBT);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector* pv;
    CellList* cl;

    real3 high, low;
    real3 targetVel;
    real kBT;

    PinnedBuffer<int> nRelevantCells{1};
    DeviceBuffer<int> relevantCells;
};

