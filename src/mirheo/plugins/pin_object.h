// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/folders.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace mirheo
{

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
    std::string ovName_;
    ObjectVector *ov_{nullptr};
    RigidObjectVector *rov_{nullptr};

    real3 translation_, rotation_;

    int reportEvery_;
    int count_{0};

    PinnedBuffer<real4> forces_, torques_;
    std::vector<char> sendBuffer_;
};

class ReportPinObjectPlugin : public PostprocessPlugin
{
public:
    ReportPinObjectPlugin(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    bool activated_;
    std::string path_;

    FileWrapper fout_;
};

} // namespace mirheo
