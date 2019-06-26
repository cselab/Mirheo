#pragma once

#include <plugins/interface.h>
#include <vector>
#include <string>
#include <core/containers.h>

#include <core/utils/folders.h>

#include <limits>
#include <cmath>

class ObjectVector;
class RigidObjectVector;

class PinObjectPlugin : public SimulationPlugin
{
public:
    constexpr static float Unrestricted = std::numeric_limits<float>::infinity();
    
    PinObjectPlugin(const MirState *state, std::string name, std::string ovName, float3 translation, float3 rotation, int reportEvery);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeIntegration(cudaStream_t stream) override;
    void afterIntegration (cudaStream_t stream) override;
    void serializeAndSend (cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string ovName;
    ObjectVector *ov;
    RigidObjectVector *rov{nullptr};

    float3 translation, rotation;

    int reportEvery;
    int count{0};

    PinnedBuffer<float4> forces, torques;
    std::vector<char> sendBuffer;
};

class ReportPinObjectPlugin : public PostprocessPlugin
{
public:
    ReportPinObjectPlugin(std::string name, std::string path);
    ~ReportPinObjectPlugin();
    
    void deserialize(MPI_Status& stat) override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    bool activated;
    std::string path;

    FILE *fout {nullptr};
    std::vector<float4> forces, torques;
};
