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


/** Compute the velocity autocorrelation function (VACF) of a given ParticleVector.
    The VACF is computed every dumpEvery steps on the time interval [startTime, endTime].
 */
class VacfPlugin : public SimulationPlugin
{
public:
    /** Create a VacfPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector from which to measure the VACF.
        \param [in] startTime VACF will use this time as origin.
        \param [in] endTime The VACF will be reported only on [startTime, endTime].
        \param [in] dumpEvery Will send the VACF to the postprocess side every this number of steps, only during the valid time interval.
    */
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


/** Postprocess side of VacfPlugin.
    Dumps the VACF in a csv file.
*/
class VacfDumper : public PostprocessPlugin
{
public:
    /** Create a VacfDumper object.
        \param [in] name The name of the plugin.
        \param [in] path The folder that will contain the vacf csv file.
    */
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
