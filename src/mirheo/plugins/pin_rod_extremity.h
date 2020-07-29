// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <string>

namespace mirheo
{

class RodVector;

/** Add alignment force on a rod segment.
 */
class PinRodExtremityPlugin : public SimulationPlugin
{
public:
    /** Create a PinRodExtremityPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] rvName The name of the RodVector to which the force should be applied.
        \param [in] segmentId The segment that will be constrained.
        \param [in] fmagn The force coefficient.
        \param [in] targetDirection The target direction.
    */
    PinRodExtremityPlugin(const MirState *state, std::string name, std::string rvName,
                          int segmentId, real fmagn, real3 targetDirection);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string rvName_;
    RodVector *rv_;
    int segmentId_;
    real fmagn_;
    real3 targetDirection_;
};

} // namespace mirheo
