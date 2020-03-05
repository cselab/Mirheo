#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/folders.h>

namespace mirheo
{

class ParticleVector;
class SDFBasedWall;

class WallRepulsionPlugin : public SimulationPlugin
{
public:
    WallRepulsionPlugin(const MirState *state, std::string name,
                        std::string pvName, std::string wallName,
                        real C, real h, real maxForce = 1e3_r);

    /// Load a snapshot of the plugin.
    WallRepulsionPlugin(const MirState *state, Loader& loader, const ConfigObject& config);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

    /// Create a \c ConfigObject describing the plugin state and register it in the saver.
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /// Implementation of snapshot saving. Reusable by potential derived classes.
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    std::string pvName_, wallName_;
    ParticleVector *pv_;
    SDFBasedWall *wall_ {nullptr};

    real C_, h_, maxForce_;
};

} // namespace mirheo
