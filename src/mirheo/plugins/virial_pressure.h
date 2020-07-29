// Copyright 2020 ETH Zurich. All Rights Reserved.
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


/** Compute the pressure in a given region from the virial theorem
    and send it to the VirialPressureDumper.
*/
class VirialPressurePlugin : public SimulationPlugin
{
public:
    /** Create a VirialPressurePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to add the particles to.
        \param [in] func The scalar field is negative in the region of interest and positive outside.
        \param [in] h The grid size used to discretize the field.
        \param [in] dumpEvery Will compute and send the pressure every this number of steps.
    */
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


/** Postprocess side of VirialPressurePlugin.
    Recieves and dump the virial pressure.
*/
class VirialPressureDumper : public PostprocessPlugin
{
public:
    /** Create a VirialPressureDumper.
        \param [in] name The name of the plugin.
        \param [in] path The csv file to which the data will be dumped.
    */
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
