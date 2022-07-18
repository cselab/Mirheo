// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>

namespace mirheo {

class ChainVector;

namespace rmacf_plugin {
using ReductionType = double;
} // namespace rmacf_plugin


/** Compute the Rouse modes autocorrelation function (RMACF) of a given ChainVector.
    The RMACF is computed every dumpEvery steps on the time interval [startTime, endTime].
 */
class RmacfPlugin : public SimulationPlugin
{
public:
    /** Create a RmacfPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] cvName The name of the ChainVector from which to measure the RMACF.
        \param [in] startTime RMACF will use this time as origin.
        \param [in] endTime The RMACF will be reported only on [startTime, endTime].
        \param [in] dumpEvery Will send the RMACF to the postprocess side every this number of steps, only during the valid time interval.
    */
    RmacfPlugin(const MirState *state, std::string name, std::string cvName,
                MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery);

    ~RmacfPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string _channelName(int p) const;

private:
    std::string cvName_;

    MirState::TimeType startTime_;
    MirState::TimeType endTime_;
    int dumpEvery_;
    bool needToSend_{false};

    int startStep_{-1};
    long nchains_{0};
    std::vector<PinnedBuffer<rmacf_plugin::ReductionType>> localRmacf_;
    MirState::TimeType savedTime_{0};
    std::vector<char> sendBuffer_;

    ChainVector *cv_{nullptr};
};


/** Postprocess side of RmacfPlugin.
    Dumps the RMACF in a csv file.
*/
class RmacfDumper : public PostprocessPlugin
{
public:
    /** Create a RmacfDumper object.
        \param [in] name The name of the plugin.
        \param [in] path The folder that will contain the rmacf csv file.
    */
    RmacfDumper(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path_;

    int numModes_{0};
    bool activated_ = true;
    FileWrapper fdump_;
};

} // namespace mirheo
