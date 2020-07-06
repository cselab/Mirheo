// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>

namespace mirheo
{

class ParticleVector;

namespace vacf_plugin
{
using ReductionType = double;
} // namespace vacf_plugin


/** A plugin to measure the velocity autocorrelation function (VACF)
    Given a \c ParticleVector and a time origin, will compute the VACF over time.
 */
class VacfPlugin : public SimulationPlugin
{
public:
    VacfPlugin(const MirState *state, std::string name, std::string pvName,
               MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery);

    ~VacfPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;

    MirState::TimeType startTime_;
    MirState::TimeType endTime_;
    int dumpEvery_;
    bool needToSend_{false};

    int startStep_{-1};
    long nparticles_{0};
    PinnedBuffer<vacf_plugin::ReductionType> localVacf_ {1};
    MirState::TimeType savedTime_{0};
    std::vector<char> sendBuffer_;

    std::string v0Channel_; ///< the channel name that will contain the initial velocities of particles
    ParticleVector *pv_{nullptr};
};


class VacfDumper : public PostprocessPlugin
{
public:
    VacfDumper(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path_;

    bool activated_ = true;
    FileWrapper fdump_;
};

} // namespace mirheo
