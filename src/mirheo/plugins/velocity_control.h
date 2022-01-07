// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>
#include "utils/pid.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/file_wrapper.h>

#include <vector>

namespace mirheo
{

class ParticleVector;

/** Apply a force in a box region to all particles.
    The force is controled by a PID controller that has a target mean velocity in that same region.
*/
class SimulationVelocityControl : public SimulationPlugin
{
public:
    /** Create a SimulationVelocityControl object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames The list of names of the ParticleVector to control.
        \param [in] low The lower coordinates of the control region.
        \param [in] high The upper coordinates of the control region.
        \param [in] sampleEvery Sample the velocity average every this number of steps.
        \param [in] tuneEvery Update the PID controller every this number of steps.
        \param [in] dumpEvery Send statistics of the PID to the postprocess plugin every this number of steps.
        \param [in] targetVel The target mean velocity in the region of interest.
        \param [in] Kp "Proportional" coefficient of the PID.
        \param [in] Ki "Integral" coefficient of the PID.
        \param [in] Kd "Derivative" coefficient of the PID.
    */
    SimulationVelocityControl(const MirState *state, std::string name, std::vector<std::string> pvNames,
                              real3 low, real3 high,
                              int sampleEvery, int tuneEvery, int dumpEvery,
                              real3 targetVel, real Kp, real Ki, real Kd);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeForces(cudaStream_t stream) override;
    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;

    bool needPostproc() override { return true; }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;

private:
    void _sampleOnePv(ParticleVector *pv, cudaStream_t stream);

private:
    int sampleEvery_, dumpEvery_, tuneEvery_;
    std::vector<std::string> pvNames_;
    std::vector<ParticleVector*> pvs_;

    real3 high_, low_;
    real3 currentVel_, targetVel_, force_;

    PinnedBuffer<int> nSamples_{1};
    PinnedBuffer<real3> totVel_{1};
    double3 accumulatedTotVel_;


    PidControl<real3> pid_;
    std::vector<char> sendBuffer_;
};


/** Postprocess side of SimulationVelocityControl.
    Receives and dump the PID stats to a csv file.
*/
class PostprocessVelocityControl : public PostprocessPlugin
{
public:
    /** Create a SimulationVelocityControl object.
        \param [in] name The name of the plugin.
        \param [in] filename The csv file to which the statistics will be dumped.
    */
    PostprocessVelocityControl(std::string name, std::string filename);

    void deserialize() override;

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;

private:
    std::string filename_;
    FileWrapper fdump_;
};

} // namespace mirheo
