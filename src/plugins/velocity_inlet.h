#pragma once

#include "interface.h"

#include <core/containers.h>

#include <functional>
#include <vector>
#include <string>


class ParticleVector;

class VelocityInletPlugin : public SimulationPlugin
{
public:

    using ImplicitSurfaceFunc = std::function<float(float3)>;
    using VelocityFieldFunc = std::function<float3(float3)>;
    
    VelocityInletPlugin(const YmrState *state, std::string name, std::string pvName,
                        ImplicitSurfaceFunc implicitSurface, VelocityFieldFunc velocityField,
                        float3 resolution);

    ~VelocityInletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeParticleDistribution(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    ParticleVector *pv;

    ImplicitSurfaceFunc implicitSurface;
    VelocityFieldFunc velocityField;
    float3 resolution;

    PinnedBuffer<float3> surfaceTriangles;
    PinnedBuffer<float3> surfaceVelocity;
    DeviceBuffer<float> cummulativeSum;
};






