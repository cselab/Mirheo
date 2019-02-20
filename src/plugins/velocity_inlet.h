#pragma once

#include "interface.h"

#include <core/containers.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

class ParticleVector;

class VelocityInletPlugin : public SimulationPlugin
{
public:

    using ImplicitSurfaceFunc = std::function<float(float3)>;
    using VelocityFieldFunc = std::function<float3(float3)>;
    
    VelocityInletPlugin(const YmrState *state, std::string name, std::string pvName,
                        ImplicitSurfaceFunc implicitSurface, VelocityFieldFunc velocityField,
                        float3 resolution, float numberDensity, float kBT);

    ~VelocityInletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector *pv;

    ImplicitSurfaceFunc implicitSurface;
    VelocityFieldFunc velocityField;
    float3 resolution;
    float numberDensity, kBT;

    PinnedBuffer<float3> surfaceTriangles;
    PinnedBuffer<float3> surfaceVelocity;
    DeviceBuffer<float> cumulativeFluxes, localFluxes;
    PinnedBuffer<int> nNewParticles {1};
    DeviceBuffer<int> workQueue; // contains id of triangle per new particle

    std::mt19937 gen {42};
    std::uniform_real_distribution<float> dist {0.f, 1.f};
};
