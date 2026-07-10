// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

/** Copies data from one ParticleVector into another one
 */
class CopyPVPlugin : public SimulationPlugin
{
public:
    /** Create a CopyPVPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvTargetName The name of the ParticleVector to which it will be copied.
        \param [in] pvSourceName The name of the ParticleVector from which it will be copied.
     */
    CopyPVPlugin(const MirState *state, const std::string& name, const std::string& pvTargetName, const std::string& pvSourceName);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvTargetName_;
    std::string pvSourceName_;
    ParticleVector *pvTarget_ {nullptr};
    ParticleVector *pvSource_ {nullptr};
};

} // namespace mirheo
