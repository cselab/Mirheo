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

class DensityControlPlugin : public SimulationPlugin
{
public:

    using RegionFunc = std::function<real(real3)>;

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

    struct LevelBounds
    {
        real lo, hi, space;
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




class PostprocessDensityControl : public PostprocessPlugin
{
public:
    PostprocessDensityControl(std::string name, std::string filename);

    void deserialize() override;

private:
    FileWrapper fdump_;
};

} // namespace mirheo
