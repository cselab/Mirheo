#pragma once

#include <mirheo/core/plugins.h>
#include <string>

namespace mirheo
{

class ParticleVector;

class ForceSaverPlugin : public SimulationPlugin
{
public:
    ForceSaverPlugin(const MirState *state, std::string name, std::string pvName);

    /// Load a snapshot of the plugin.
    ForceSaverPlugin(const MirState *state, Loader& loader, const ConfigObject& config);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    bool needPostproc() override;
    void beforeIntegration(cudaStream_t stream) override;

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    std::string pvName_;
    ParticleVector *pv_;
    static const std::string fieldName_;
};

} // namespace mirheo
