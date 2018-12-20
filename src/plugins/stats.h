#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>
#include <core/utils/timer.h>

class ParticleVector;
class CellList;

using ReductionType = double;

class SimulationStats : public SimulationPlugin
{
private:
    int fetchEvery;
    bool needToDump{false};

    int nparticles;
    PinnedBuffer<ReductionType> momentum{3}, energy{1};
    PinnedBuffer<float> maxvel{1};
    std::vector<char> sendBuffer;

    mTimer timer;

public:
    SimulationStats(std::string name, const YmrState *state, int fetchEvery);

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }
};

class PostprocessStats : public PostprocessPlugin
{
private:
    MPI_Datatype mpiReductionType;
    FILE *fdump = nullptr;

public:
    PostprocessStats(std::string name, std::string filename = "");
    ~PostprocessStats();

    void deserialize(MPI_Status& stat) override;
};
