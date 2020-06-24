#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>

#include <functional>
#include <vector>

namespace mirheo
{

class ParticleVector;

using FuncTime3D = std::function<std::vector<real3>(real)>;

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
    std::string pvName_;
    ParticleVector *pv_;

    FuncTime3D positions_;
    FuncTime3D velocities_;

    PinnedBuffer<double3> forces_;
    PinnedBuffer<real3> posBuffer_, velBuffer_;
    PinnedBuffer<int> pids_;

    int nsamples_ {0};
    int reportEvery_;
    std::vector<char> sendBuffer_;
};




class AnchorParticlesStatsPlugin : public PostprocessPlugin
{
public:
    AnchorParticlesStatsPlugin(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    bool activated_;
    std::string path_;

    FileWrapper fout_;
};

} // namespace mirheo
