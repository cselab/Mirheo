#pragma once

#include <mirheo/plugins/interface.h>
#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/timer.h>

namespace mirheo
{

class ParticleVector;
class SDF_basedWall;

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
    int sampleEvery, dumpEvery;
    int nsamples {0};
    
    std::string wallName;
    std::string frozenPvName;
    
    bool needToDump {false};

    SDF_basedWall *wall;
    ParticleVector *pv;
    
    PinnedBuffer<double3> *bounceForceBuffer {nullptr};
    PinnedBuffer<double3> pvForceBuffer {1};
    double3 totalForce {0.0, 0.0, 0.0};
    
    std::vector<char> sendBuffer;
};

class WallForceDumperPlugin : public PostprocessPlugin
{
public:
    WallForceDumperPlugin(std::string name, std::string filename);

    void deserialize() override;

private:
    FileWrapper fdump;
};

} // namespace mirheo
