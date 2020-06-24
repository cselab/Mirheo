#pragma once

#include <mirheo/core/plugins.h>
#include "utils/pid.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/file_wrapper.h>

#include <vector>

namespace mirheo
{

class ParticleVector;

class SimulationVelocityControl : public SimulationPlugin
{
public:
    SimulationVelocityControl(const MirState *state, std::string name, std::vector<std::string> pvNames,
                              real3 low, real3 high,
                              int sampleEvery, int tuneEvery, int dumpEvery,
                              real3 targetVel, real Kp, real Ki, real Kd);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;

private:
    void _sampleOnePv(ParticleVector *pv, cudaStream_t stream);

private:
    int sampleEvery_, dumpEvery_, tuneEvery_;
    std::vector<std::string> pvNames_;
    std::vector<ParticleVector*> pvs_;

    real3 high_, low_;
    real3 currentVel_, targetVel_, force_;

    PinnedBuffer<int> nSamples_{1};
    PinnedBuffer<real3> totVel_{1};
    double3 accumulatedTotVel_;


    PidControl<real3> pid_;
    std::vector<char> sendBuffer_;
};

class PostprocessVelocityControl : public PostprocessPlugin
{
public:
    PostprocessVelocityControl(std::string name, std::string filename);

    void deserialize() override;

private:
    FileWrapper fdump_;
};

} // namespace mirheo
