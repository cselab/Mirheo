// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/file_wrapper.h>


namespace mirheo
{

struct StressTensor{
    real p[9]; //"Pxx", "Pxy", "Pxz", "Pyx", "Pyy", "Pyz", "Pzx", "Pzy", "Pzz"
};

class ParticleVector;

namespace stress_tensor_plugin
{
using ReductionType = StressTensor;
} // namespace stress_tensor_plugin


/** Compute the stress tensor in the system
    and send it to the StressTensorDumper.
*/
class StressTensorPlugin : public SimulationPlugin
{
public:
    /** Create a StressTensorPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to add the particles to.
        \param [in] dumpEvery Will compute and send the stress every this number of steps.
    */
    StressTensorPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery);
    ~StressTensorPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void beforeIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName_;
    int dumpEvery_;
    

    bool needToSend_ = false;

    PinnedBuffer<stress_tensor_plugin::ReductionType> localStressTensor_ {1};
    MirState::TimeType savedTime_ = 0;

    std::vector<char> sendBuffer_;

    ParticleVector *pv_;
};


/** Postprocess side of StressTensorPlugin.
    Recieves and dump the stress tensor.
*/
class StressTensorDumper : public PostprocessPlugin
{
public:
    /** Create a StressTensorDumper.
        \param [in] name The name of the plugin.
        \param [in] path The csv file to which the data will be dumped.
    */
    StressTensorDumper(std::string name, std::string mask, std::string path);

    void deserialize() override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path_;
    std::string fname_;

    std::string mask_;
    bool bool_mask[9];
    const char *stress_label[9] = {"Pxx", "Pxy", "Pxz", "Pyx", "Pyy", "Pyz", "Pzx", "Pzy", "Pzz"};
    std::string comment_;

    bool activated_ = true;
    FileWrapper fdump_;
};

} // namespace mirheo
