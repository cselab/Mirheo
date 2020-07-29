// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/path.h>

namespace mirheo
{

class ParticleVector;

/** Add or set maxwellian drawn velocities to the particles of a given ParticleVector.
 */
class TemperaturizePlugin : public SimulationPlugin
{
public:
    /** Create a TemperaturizePlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to modify.
        \param [in] kBT Target temperature.
        \param [in] keepVelocity Wether to add or reset the velocities.
    */
    TemperaturizePlugin(const MirState *state, std::string name, std::string pvName, real kBT, bool keepVelocity);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_;
    ParticleVector *pv_;
    real kBT_;
    bool keepVelocity_;
};

} // namespace mirheo
