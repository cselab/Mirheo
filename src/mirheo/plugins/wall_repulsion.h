#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/utils/folders.h>

namespace mirheo
{

class ParticleVector;
class SDF_basedWall;

class WallRepulsionPlugin : public SimulationPlugin
{
public:
    WallRepulsionPlugin(const MirState *state, std::string name,
                        std::string pvName, std::string wallName,
                        real C, real h, real maxForce = 1e3_r);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName_, wallName_;
    ParticleVector *pv_;
    SDF_basedWall *wall_;

    real C_, h_, maxForce_;
};

} // namespace mirheo
