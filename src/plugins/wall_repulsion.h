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
                        float C, float h, float maxForce = 1e3f);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName, wallName;
    ParticleVector* pv;
    SDF_basedWall *wall;

    float C, h, maxForce;
};

