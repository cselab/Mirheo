// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <functional>
#include <random>

namespace mirheo
{

class ParticleVector;

/** Add particles to a given ParticleVector.
    The particles are injected on a given surface at a given influx rate.
 */
class VelocityInletPlugin : public SimulationPlugin
{
public:

    /// Representation of a surface from a scalar field.
    using ImplicitSurfaceFunc = std::function<real(real3)>;

    /// Velocity field used to describe the inflow.
    using VelocityFieldFunc = std::function<real3(real3)>;

    /** Create a VelocityInletPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvName The name of the ParticleVector to add the particles to.
        \param [in] implicitSurface The scalar field that has the desired surface as zero level set.
        \param [in] velocityField The inflow velocity. Only relevant on the surface.
        \param [in] resolution Grid size used to sample the fields.
        \param [in] numberDensity The target number density of injection.
        \param [in] kBT The temperature of the injected particles.
    */
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
