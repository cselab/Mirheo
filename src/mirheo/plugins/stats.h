#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/timer.h>

namespace mirheo
{

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
    void saveSnapshotAndRegister(Dumper& dumper) override;

protected:
    ConfigDictionary _saveSnapshot(Dumper& dumper, const std::string& typeName);

private:
    int fetchEvery_;
    bool needToDump_{false};

    Stats::CountType nparticles_;
    PinnedBuffer<Stats::ReductionType> momentum_{3}, energy_{1};
    PinnedBuffer<real> maxvel_{1};
    std::vector<char> sendBuffer_;

    std::vector<ParticleVector*> pvs_;

    mTimer timer_;
};

class PostprocessStats : public PostprocessPlugin
{
public:
    PostprocessStats(std::string name, std::string filename = "");

    void deserialize() override;
    void saveSnapshotAndRegister(Dumper& dumper) override;

protected:
    ConfigDictionary _saveSnapshot(Dumper& dumper, const std::string& typeName);

private:
    FileWrapper fdump_;
    std::string filename_;
};

} // namespace mirheo
