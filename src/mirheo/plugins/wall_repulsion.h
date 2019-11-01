#pragma once

#include <plugins/interface.h>
#include <core/containers.h>
#include <string>

#include <core/utils/folders.h>

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
    std::string pvName, wallName;
    ParticleVector* pv;
    SDF_basedWall *wall;

    real C, h, maxForce;
};

