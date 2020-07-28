// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/plugins.h>

#include <string>
#include <vector>

namespace mirheo
{

class ParticleVector;

/** Add a constant to the velocity of particles in a given region such that it matches a given average.
*/
class ImposeVelocityPlugin : public SimulationPlugin
{
public:
    /** Create a ImposeVelocityPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames The name of the (list of) ParticleVector to modify.
        \param [in] low Lower coordinates of the region of interest.
        \param [in] high Upper coordinates of the region of interest.
        \param [in] targetVel The target average velocity in the region.
        \param [in] every Correct the velocity every this number of time steps.
     */
    ImposeVelocityPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                         real3 low, real3 high, real3 targetVel, int every);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

    /** Change the target velocity to a new value.
        \param [in] v The new target velocity.
    */
    void setTargetVelocity(real3 v);

private:
    std::vector<std::string> pvNames_;
    std::vector<ParticleVector*> pvs_;

    real3 high_, low_;
    real3 targetVel_;

    int every_;

    PinnedBuffer<int> nSamples_{1};
    PinnedBuffer<double3> totVel_{1};
};

} // namespace mirheo
