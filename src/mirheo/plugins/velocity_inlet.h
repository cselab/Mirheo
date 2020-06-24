#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <functional>
#include <random>

namespace mirheo
{

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
    std::string pvName_;
    ParticleVector *pv_;

    ImplicitSurfaceFunc implicitSurface_;
    VelocityFieldFunc velocityField_;
    real3 resolution_;
    real numberDensity_, kBT_;

    PinnedBuffer<real3> surfaceTriangles_;
    PinnedBuffer<real3> surfaceVelocity_;
    DeviceBuffer<real> cumulativeFluxes_, localFluxes_;
    PinnedBuffer<int> nNewParticles_ {1};
    DeviceBuffer<int> workQueue_; // contains id of triangle per new particle

    std::mt19937 gen_ {42};
    std::uniform_real_distribution<real> dist_ {0._r, 1._r};
};

} // namespace mirheo
