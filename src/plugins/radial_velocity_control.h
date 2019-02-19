#pragma once

#include "interface.h"
#include "pid.h"

#include <core/containers.h>
#include <core/datatypes.h>

#include <vector>

class ParticleVector;

class SimulationRadialVelocityControl : public SimulationPlugin
{
public:
    SimulationRadialVelocityControl(const YmrState *state, std::string name, std::vector<std::string> pvNames,
                                    float minRadius, float maxRadius, int sampleEvery, int tuneEvery, int dumpEvery,
                                    float3 center, float targetVel, float Kp, float Ki, float Kd);

    ~SimulationRadialVelocityControl();
    
    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    int sampleEvery, dumpEvery, tuneEvery;
    std::vector<std::string> pvNames;
    std::vector<ParticleVector*> pvs;

    float currentVel, targetVel, force;
    float minRadiusSquare, maxRadiusSquare;
    float3 center;

    PinnedBuffer<int> nSamples{1};
    PinnedBuffer<double> totVel{1};
    double accumulatedTotVel;
    

    PidControl<float> pid;
    std::vector<char> sendBuffer;

private:
    void sampleOnePv(ParticleVector *pv, cudaStream_t stream);
};

class PostprocessRadialVelocityControl : public PostprocessPlugin
{
private:
    FILE *fdump;

public:
    PostprocessRadialVelocityControl(std::string name, std::string filename);
    ~PostprocessRadialVelocityControl();

    void deserialize(MPI_Status& stat) override;
};
