#pragma once

#include <plugins/interface.h>
#include <plugins/pid.h>
#include <core/containers.h>
#include <core/datatypes.h>

class ParticleVector;
class CellList;

class SimulationVelocityControl : public SimulationPlugin
{
public:
    SimulationVelocityControl(std::string name, std::string pvName,
                              float3 low, float3 high, int sampleEvery, int dumpEvery, 
                              float3 targetVel, float Kp, float Ki, float Kd) :
        SimulationPlugin(name), pvName(pvName), low(low), high(high),
        currentVel(make_float3(0,0,0)), targetVel(targetVel),
        dumpEvery(dumpEvery), sampleEvery(sampleEvery),
        force(make_float3(0, 0, 0)),
        pid(make_float3(0, 0, 0), Kp, Ki, Kd)
    {}

    void setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

private:
    int sampleEvery, dumpEvery;
    std::string pvName;
    ParticleVector* pv;

    float3 high, low;
    float3 currentVel, targetVel, force;

    PinnedBuffer<int> nSamples{1};
    PinnedBuffer<float3> totVel{1};

    PidControl<float3> pid;
    std::vector<char> sendBuffer;
};

class PostprocessVelocityControl : public PostprocessPlugin
{
private:
    MPI_Datatype mpiReductionType;
    FILE *fdump;

public:
    PostprocessVelocityControl(std::string name, std::string filename);
    ~PostprocessVelocityControl();

    void deserialize(MPI_Status& stat) override;
};
