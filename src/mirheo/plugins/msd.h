// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>

namespace mirheo
{

class ParticleVector;

namespace msd_plugin
{
using ReductionType = double;
} // namespace msd_plugin


/** Compute the mean squared distance (MSD) of a given ParticleVector.
    The MSD is computed every dumpEvery steps on the time interval [startTime, endTime].

    Each particle stores the total displacement from startTime.
    To compute this, it also stores its position at each step.
 */
class MsdPlugin : public SimulationPlugin
{
public:
    /** Create a MsdPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector from which to measure the MSD.
        \param [in] startTime MSD will use this time as origin.
        \param [in] endTime The MSD will be reported only on [startTime, endTime].
        \param [in] dumpEvery Will send the MSD to the postprocess side every this number of steps, only during the valid time interval.
    */
    MsdPlugin(const MirState *state, std::string name, std::string pvName,
              MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery);

    ~MsdPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    ParticleVector *pv_;

    MirState::TimeType startTime_;
    MirState::TimeType endTime_;
    int dumpEvery_;
    bool needToSend_{false};

    int startStep_{-1};
    long nparticles_{0};
    PinnedBuffer<msd_plugin::ReductionType> localMsd_ {1};
    MirState::TimeType savedTime_{0};
    std::vector<char> sendBuffer_;

    std::string previousPositionChannelName_;  ///< Name of the channel that will contain the previous positions of the particles
    std::string totalDisplacementChannelName_; ///< Name of the channel that will contain the total displacements of the particles
};


/** Postprocess side of MsdPlugin.
    Dumps the VACF in a csv file.
*/
class MsdDumper : public PostprocessPlugin
{
public:
    /** Create a MsdDumper object.
        \param [in] name The name of the plugin.
        \param [in] path The folder that will contain the vacf csv file.
    */
    MsdDumper(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path_;

    bool activated_ = true;
    FileWrapper fdump_;
};

} // namespace mirheo
