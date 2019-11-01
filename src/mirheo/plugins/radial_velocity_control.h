#pragma once

#include "interface.h"
#include "utils/pid.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/file_wrapper.h>

#include <vector>

class ParticleVector;

class SimulationRadialVelocityControl : public SimulationPlugin
{
public:
    SimulationRadialVelocityControl(const MirState *state, std::string name, std::vector<std::string> pvNames,
                                    real minRadius, real maxRadius, int sampleEvery, int tuneEvery, int dumpEvery,
                                    real3 center, real targetVel, real Kp, real Ki, real Kd);

    ~SimulationRadialVelocityControl();
    
    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart(MPI_Comm comm, const std::string& path) override;

private:
    int sampleEvery, dumpEvery, tuneEvery;
    std::vector<std::string> pvNames;
    std::vector<ParticleVector*> pvs;

    real currentVel, targetVel, force;
    real minRadiusSquare, maxRadiusSquare;
    real3 center;

    PinnedBuffer<int> nSamples{1};
    PinnedBuffer<double> totVel{1};
    long double        accumulatedVel;
    unsigned long long accumulatedSamples;

    PidControl<real> pid;
    std::vector<char> sendBuffer;

private:
    void sampleOnePv(ParticleVector *pv, cudaStream_t stream);
};

class PostprocessRadialVelocityControl : public PostprocessPlugin
{
public:
    PostprocessRadialVelocityControl(std::string name, std::string filename);

    void deserialize() override;

private:
    FileWrapper fdump;
};
