#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <core/datatypes.h>
#include <core/utils/file_wrapper.h>
#include <core/utils/timer.h>

class ParticleVector;

namespace Stats
{
using ReductionType = double;
using CountType = unsigned long long;
}

class SimulationStats : public SimulationPlugin
{
public:
    SimulationStats(const MirState *state, std::string name, int fetchEvery);
    ~SimulationStats();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    int fetchEvery;
    bool needToDump{false};

    Stats::CountType nparticles;
    PinnedBuffer<Stats::ReductionType> momentum{3}, energy{1};
    PinnedBuffer<real> maxvel{1};
    std::vector<char> sendBuffer;

    std::vector<ParticleVector*> pvs;

    mTimer timer;
};

class PostprocessStats : public PostprocessPlugin
{
public:
    PostprocessStats(std::string name, std::string filename = "");

    void deserialize() override;

private:
    FileWrapper fdump;
};
