#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/utils/file_wrapper.h>
#include <core/utils/folders.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

class ObjectVector;
class RigidObjectVector;

class PinObjectPlugin : public SimulationPlugin
{
public:
    constexpr static real Unrestricted = std::numeric_limits<real>::infinity();
    
    PinObjectPlugin(const MirState *state, std::string name, std::string ovName, real3 translation, real3 rotation, int reportEvery);

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

    real3 translation, rotation;

    int reportEvery;
    int count{0};

    PinnedBuffer<real4> forces, torques;
    std::vector<char> sendBuffer;
};

class ReportPinObjectPlugin : public PostprocessPlugin
{
public:
    ReportPinObjectPlugin(std::string name, std::string path);
    
    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    bool activated;
    std::string path;

    FileWrapper fout;
    std::vector<real4> forces, torques;
};
