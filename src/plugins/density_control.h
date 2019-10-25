#pragma once

#include "interface.h"
#include "utils/pid.h"

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/utils/file_wrapper.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

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

    int sampleEvery, dumpEvery, tuneEvery;
    std::vector<std::string> pvNames;
    std::vector<ParticleVector*> pvs;

    LevelBounds levelBounds;
    real targetDensity;
    
    std::unique_ptr<Field> spaceDecompositionField; /// a scalar field used to decompose the space with level sets

    int nSamples;                                   /// number of times we called sample function
    PinnedBuffer<unsigned long long int> nInsides;  /// number of samples per subregion
    std::vector<double> volumes;                    /// volume of each subregion

    std::vector<real> densities;
    PinnedBuffer<real> forces;
    
    std::vector<PidControl<real>> controllers;
    real Kp, Ki, Kd;

    std::vector<char> sendBuffer;
private:

    void computeVolumes(cudaStream_t stream, int MCnSamples);
    void sample(cudaStream_t stream);
    void updatePids(cudaStream_t stream);
    void applyForces(cudaStream_t stream);
};




class PostprocessDensityControl : public PostprocessPlugin
{
public:
    PostprocessDensityControl(std::string name, std::string filename);

    void deserialize() override;

private:
    FileWrapper fdump;
};
