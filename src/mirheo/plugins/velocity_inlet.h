#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

class ParticleVector;

class VelocityInletPlugin : public SimulationPlugin
{
public:

    using ImplicitSurfaceFunc = std::function<real(real3)>;
    using VelocityFieldFunc = std::function<real3(real3)>;
    
    VelocityInletPlugin(const MirState *state, std::string name, std::string pvName,
                        ImplicitSurfaceFunc implicitSurface, VelocityFieldFunc velocityField,
                        real3 resolution, real numberDensity, real kBT);

    ~VelocityInletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector *pv;

    ImplicitSurfaceFunc implicitSurface;
    VelocityFieldFunc velocityField;
    real3 resolution;
    real numberDensity, kBT;

    PinnedBuffer<real3> surfaceTriangles;
    PinnedBuffer<real3> surfaceVelocity;
    DeviceBuffer<real> cumulativeFluxes, localFluxes;
    PinnedBuffer<int> nNewParticles {1};
    DeviceBuffer<int> workQueue; // contains id of triangle per new particle

    std::mt19937 gen {42};
    std::uniform_real_distribution<real> dist {0._r, 1._r};
};
