// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace mirheo
{

class ParticleVector;
class CellList;
class ScalarField;

/** Base class for outlet Plugins.
    Outlet plugins delete particles of given a ParticleVector list in a region.
 */
class OutletPlugin : public SimulationPlugin
{
public:
    /** Create a OutletPlugin.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames List of names of the ParticleVector that the outlet will be applied to.
     */
    OutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames);
    ~OutletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override { return false; }

protected:

    std::vector<std::string> pvNames_; ///< The ParticleVector names
    std::vector<ParticleVector*> pvs_; ///< List of ParticleVector that will have its particles removed.

    DeviceBuffer<int> nParticlesInside_ {1}; ///< Current number of particles in the outlet region.

    std::mt19937 gen_ {42}; ///< helper RNG to get a new random seed at each time step.
    std::uniform_real_distribution<real> udistr_ {0._r, 1._r};  ///< helper to get a float random seed at each time step in [0, 1].
};


/** Delete all particles that cross a given plane.
 */
class PlaneOutletPlugin : public OutletPlugin
{
public:
    /** Create a PlaneOutletPlugin.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames List of names of the ParticleVector that the outlet will be applied to.
        \param [in] plane Coefficients (a, b, c, d) of the plane.

        A particle crosses the plane if a*x + b*y + c*z + d goes from nbegative to postive accross one time step.
     */
    PlaneOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames, real4 plane);

    ~PlaneOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;

private:
    real4 plane_;
};

/** Delete all particles in a given region, defined implicitly by a field.
A particle is considered inside the region if the given field is negative at the particle's position.
 */
class RegionOutletPlugin : public OutletPlugin
{
public:
    /// A scalar field to represent inside (negative) / outside (positive) region
    using RegionFunc = std::function<real(real3)>;

    /** Create a RegionOutletPlugin.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames List of names of the ParticleVector that the outlet will be applied to.
        \param [in] region The field that describes the region. This will be sampled on a uniform grid and uploaded to the GPU.
        \param [in] resolution The grid space used to discretize \p region.
     */
    RegionOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                       RegionFunc region, real3 resolution);

    ~RegionOutletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override { return false; }

protected:

    /** Compute the volume inside The outlet region using Monte Carlo.
        \param [in] nSamples The number of samples per rank.
        \param [in] seed Random seed.
        \return The estimated volume.
    */
    double _computeVolume(long long int nSamples, real seed) const;

    /** Count the current number of particles that are inside the outlet region.
        The result is stored in the **device** array of nParticlesInside_.
        \param [in] stream The compute stream.
     */
    void _countInsideParticles(cudaStream_t stream);

protected:
    double volume_; ///< Volume estimate of the outlet region.
    std::unique_ptr<ScalarField> outletRegion_; ///< Scalar field that describes the outlet region.
};

/** Delete particles located in a given region if the number density is higher than a target one.
 */
class DensityOutletPlugin : public RegionOutletPlugin
{
public:
    /** Create a DensityOutletPlugin.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames List of names of the ParticleVector that the outlet will be applied to.
        \param [in] numberDensity The target number density.
        \param [in] region The field that describes the region. This will be sampled on a uniform grid and uploaded to the GPU.
        \param [in] resolution The grid space used to discretize \p region.
     */
    DensityOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                        real numberDensity, RegionFunc region, real3 resolution);
    ~DensityOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;

private:
    real numberDensity_;
};


/** Delete particles located in a given region at a given rate.
 */
class RateOutletPlugin : public RegionOutletPlugin
{
public:
    /** Create a RateOutletPlugin.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames List of names of the ParticleVector that the outlet will be applied to.
        \param [in] rate The rate of deletion of particles.
        \param [in] region The field that describes the region. This will be sampled on a uniform grid and uploaded to the GPU.
        \param [in] resolution The grid space used to discretize \p region.
     */
    RateOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                     real rate, RegionFunc region, real3 resolution);
    ~RateOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;

private:
    real rate_;
};

} // namespace mirheo
