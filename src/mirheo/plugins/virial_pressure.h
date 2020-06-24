#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/field/from_function.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>

namespace mirheo
{

class ParticleVector;

namespace virial_pressure_plugin
{
using ReductionType = double;
} // namespace virial_pressure_plugin

class VirialPressurePlugin : public SimulationPlugin
{
public:
    VirialPressurePlugin(const MirState *state, std::string name, std::string pvName,
                         FieldFunction func, real3 h, int dumpEvery);

    ~VirialPressurePlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    int dumpEvery_;
    bool needToSend_ = false;

    FieldFromFunction region_;

    PinnedBuffer<virial_pressure_plugin::ReductionType> localVirialPressure_ {1};
    MirState::TimeType savedTime_ = 0;

    std::vector<char> sendBuffer_;

    ParticleVector *pv_;
};


class VirialPressureDumper : public PostprocessPlugin
{
public:
    VirialPressureDumper(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path_;

    bool activated_ = true;
    FileWrapper fdump_;
};

} // namespace mirheo
