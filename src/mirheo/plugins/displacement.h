// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

#include <string>

namespace mirheo
{

class ParticleVector;

/** Compute the dispacement of particles between a given number of time steps.
 */
class ParticleDisplacementPlugin : public SimulationPlugin
{
public:
    /** Create a ParticleDisplacementPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the concerned ParticleVector.
        \param [in] updateEvery The number of steps between two steps used to compute the displacement.
     */
    ParticleDisplacementPlugin(const MirState *state, std::string name, std::string pvName, int updateEvery);
    ~ParticleDisplacementPlugin();

    void afterIntegration(cudaStream_t stream) override;

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override {return false;}

private:
    std::string pvName_;
    ParticleVector *pv_;
    int updateEvery_;

    static const std::string displacementChannelName_;
    static const std::string savedPositionChannelName_;
};

} // namespace mirheo
