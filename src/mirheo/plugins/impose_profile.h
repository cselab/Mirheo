// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/path.h>

#include <string>
#include <vector>

namespace mirheo
{

class ParticleVector;
class CellList;

/** Set the velocity to a given one in a box.
    The velocity is set to a constant plus a random velocity that has a Maxwell distribution.
*/
class ImposeProfilePlugin : public SimulationPlugin
{
public:
    /** Create a ImposeProfilePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to modify.
        \param [in] low Lower coordinates of the region of interest.
        \param [in] high Upper coordinates of the region of interest.
        \param [in] targetVel The constant part of the new velocity.
        \param [in] kBT Temperature used to draw the velocity from the maxwellian distribution.
     */
    ImposeProfilePlugin(const MirState *state, std::string name, std::string pvName,
                        real3 low, real3 high, real3 targetVel, real kBT);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    bool needPostproc() override { return false; }

    void afterIntegration(cudaStream_t stream) override;

private:
    std::string pvName_;
    ParticleVector *pv_;
    CellList *cl_;

    real3 high_, low_;
    real3 targetVel_;
    real kBT_;

    PinnedBuffer<int> nRelevantCells_{1};
    DeviceBuffer<int> relevantCells_;
};

} // namespace mirheo
