#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

namespace mirheo
{

class ParticleVector;

class BerendsenThermostatPlugin : public SimulationPlugin
{
public:
    BerendsenThermostatPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                              real kBT, real tau, bool increaseIfLower);

    /// Load the plugin from a snasphot.
    BerendsenThermostatPlugin(const MirState *state, Loader& loader, const ConfigObject& config);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

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
