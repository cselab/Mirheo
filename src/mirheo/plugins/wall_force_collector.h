#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/timer.h>

namespace mirheo
{

class ParticleVector;
class SDFBasedWall;

class WallForceCollectorPlugin : public SimulationPlugin
{
public:
    WallForceCollectorPlugin(const MirState *state, std::string name,
                             std::string wallName, std::string frozenPvName,
                             int sampleEvery, int dumpEvery);
    ~WallForceCollectorPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    int sampleEvery_, dumpEvery_;
    int nsamples_ {0};
    
    std::string wallName_;
    std::string frozenPvName_;
    
    bool needToDump_ {false};

    SDFBasedWall *wall_;
    ParticleVector *pv_;
    
    PinnedBuffer<double3> *bounceForceBuffer_ {nullptr};
    PinnedBuffer<double3> pvForceBuffer_ {1};
    double3 totalForce_ {0.0, 0.0, 0.0};
    
    std::vector<char> sendBuffer_;
};

class WallForceDumperPlugin : public PostprocessPlugin
{
public:
    WallForceDumperPlugin(std::string name, std::string filename);

    void deserialize() override;

private:
    FileWrapper fdump_;
};

} // namespace mirheo
