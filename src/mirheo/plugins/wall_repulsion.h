// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/path.h>

namespace mirheo
{

class ParticleVector;
class SDFBasedWall;


/** Add a force that pushes particles away from the wall surfaces.
    The magnitude of the force decreases linearly down to zero at a given distance h.
    Furthermore, the force can be capped.
 */
class WallRepulsionPlugin : public SimulationPlugin
{
public:
    /** Create a WallRepulsionPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector that will be subject to the force.
        \param [in] wallName The name of the \c Wall.
        \param [in] C Force coefficient.
        \param [in] h Force maximum distance.
        \param [in] maxForce Maximum force magnitude.
    */
    WallRepulsionPlugin(const MirState *state, std::string name,
                        std::string pvName, std::string wallName,
                        real C, real h, real maxForce);

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
