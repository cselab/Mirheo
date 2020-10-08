// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/plugins.h>
#include "utils/pid.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/file_wrapper.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mirheo
{

class ParticleVector;
class Field;

/** Apply forces on particles in order to keep the number density constant within layers in a field.
    The layers are determined by the level sets of the field.
    Forces are perpendicular to these layers; their magnitude is computed from PID controllers.

    Cannot be used with multiple invocations of `Mirheo.run`.
 */
class DensityControlPlugin : public SimulationPlugin
{
public:
    /// functor that describes the region in terms of level sets.
    using RegionFunc = std::function<real(real3)>;

    /** Create a DensityControlPlugin object.
        \param [in] state The global state of the simulation.
        \param [in] name The name of the plugin.
        \param [in] pvNames The names of the ParticleVector that have the target density..
        \param [in] targetDensity The target number density.
        \param [in] region The field used to partition the space.
        \param [in] resolution The grid spacing used to discretized \p region
        \param [in] levelLo The minimum level set of the region to control.
        \param [in] levelHi The maximum level set of the region to control.
        \param [in] levelSpace Determines the difference between 2 consecutive layers in the partition of space.
        \param [in] Kp "Proportional" coefficient of the PID.
        \param [in] Ki "Integral" coefficient of the PID.
        \param [in] Kd "Derivative" coefficient of the PID.
        \param [in] tuneEvery Update th PID controllers every this number of steps.
        \param [in] dumpEvery Dump statistics every this number of steps. See also PostprocessDensityControl.
        \param [in] sampleEvery Sample statistics every this number of steps. Used by PIDs.
     */
    DensityControlPlugin(const MirState *state, std::string name,
                         std::vector<std::string> pvNames, real targetDensity,
                         RegionFunc region, real3 resolution,
                         real levelLo, real levelHi, real levelSpace,
                         real Kp, real Ki, real Kd,
                         int tuneEvery, int dumpEvery, int sampleEvery);

    ~DensityControlPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    bool needPostproc() override { return true; }

    /// Helper structure to partition the space.
    struct LevelBounds
    {
        real lo; ///< Smallest level set.
        real hi; ///< Largest level set.
        real space; ///< Difference between two level sets.
    };

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;

private:
    void computeVolumes(cudaStream_t stream, int MCnSamples);
    void sample(cudaStream_t stream);
    void updatePids(cudaStream_t stream);
    void applyForces(cudaStream_t stream);

private:
    int sampleEvery_, dumpEvery_, tuneEvery_;
    std::vector<std::string> pvNames_;
    std::vector<ParticleVector*> pvs_;

    LevelBounds levelBounds_;
    real targetDensity_;

    std::unique_ptr<Field> spaceDecompositionField_; /// a scalar field used to decompose the space with level sets

    int nSamples_;                                   /// number of times we called sample function
    PinnedBuffer<unsigned long long int> nInsides_;  /// number of samples per subregion
    std::vector<double> volumes_;                    /// volume of each subregion

    std::vector<real> densities_;
    PinnedBuffer<real> forces_;

    std::vector<PidControl<real>> controllers_;
    real Kp_, Ki_, Kd_;

    std::vector<char> sendBuffer_;
};



/** Postprocessing side of DensityControlPlugin.
    Dumps the density and force in each layer of the space partition.
 */
class PostprocessDensityControl : public PostprocessPlugin
{
public:
    /** Create a PostprocessDensityControl object.
        \param [in] name The name of the plugin.
        \param [in] filename The txt file that will contain the density and corresponding force magnitudes in each layer.
     */
    PostprocessDensityControl(std::string name, std::string filename);

    void deserialize() override;

private:
    FileWrapper fdump_;
};

} // namespace mirheo
