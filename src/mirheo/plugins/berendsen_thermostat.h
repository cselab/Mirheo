// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

/** Apply Berendsen thermostat to the given particles.
 */
class BerendsenThermostatPlugin : public SimulationPlugin
{
public:
    /** Create a BerendsenThermostatPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames The list of names of the concerned ParticleVector s.
        \param [in] kBT The target temperature, in energy units.
        \param [in] tau The relaxation time.
        \param [in] increaseIfLower Whether to increase the temperature if it’s lower than the target temperature.
     */
    BerendsenThermostatPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                              real kBT, real tau, bool increaseIfLower);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::vector<std::string> pvNames_;
    std::vector<ParticleVector *> pvs_;
    real kBT_;
    real tau_;
    bool increaseIfLower_;

    /// Sum of (m * vx, m * vy, m * vz, m * v^2).
    PinnedBuffer<real4> stats_;
};

} // namespace mirheo
