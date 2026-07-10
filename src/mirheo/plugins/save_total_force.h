// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>

namespace mirheo
{

struct totForce {
    real x = 0.0;
    real y = 0.0;
    real z = 0.0;
    //real dummy; //is this useful for cache alignment?
};

class ParticleVector;

namespace total_force_saver_plugin
{
using ReductionType = totForce;
} // namespace total_force_saver_plugin


/** Compute the total force in the x,y,z-direction in the system
    and send it to the TotalForceSaverDumper.
*/
class TotalForceSaverPlugin : public SimulationPlugin
{
public:
    /** Create a TotalForceSaverPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to add the particles to.
        \param [in] dumpEvery Will compute and send the total force every this number of steps.
    */
    TotalForceSaverPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery);
    ~TotalForceSaverPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    int dumpEvery_;
    bool needToSend_ = false;

    PinnedBuffer<total_force_saver_plugin::ReductionType> localForce_ {1};
    MirState::TimeType savedTime_ = 0;

    std::vector<char> sendBuffer_;

    ParticleVector *pv_;
};


/** Postprocess side of TotalForceSaverPlugin.
    Recieves and dump the total force.
*/
class TotalForceSaverDumper : public PostprocessPlugin
{
public:
    /** Create a TotalForceSaverDumper.
        \param [in] name The name of the plugin.
        \param [in] path The csv file to which the data will be dumped.
    */
    TotalForceSaverDumper(std::string name, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path_;
    std::string fname_;

    bool activated_ = true;
    FileWrapper fdump_;
};

} // namespace mirheo
