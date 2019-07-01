#pragma once

#include "interface.h"
#include "utils/pid.h"

#include <core/containers.h>
#include <core/datatypes.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

class ParticleVector;
class Field;

class DensityControlPlugin : public SimulationPlugin
{
public:

    using RegionFunc = std::function<float(float3)>;
    
    DensityControlPlugin(const MirState *state, std::string name,
                         std::vector<std::string> pvNames, float targetDensity,
                         RegionFunc region, float3 resolution,
                         float levelLo, float levelHi, float levelSpace,
                         float Kp, float Ki, float Kd,
                         int tuneEvery, int dumpEvery, int sampleEvery);
    
    ~DensityControlPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    bool needPostproc() override { return true; }

    struct LevelBounds
    {
        float lo, hi, space;
    };

    void checkpoint(MPI_Comm comm, std::string path, int checkpointId) override;
    void restart(MPI_Comm comm, std::string path) override;
    
private:

    int sampleEvery, dumpEvery, tuneEvery;
    std::vector<std::string> pvNames;
    std::vector<ParticleVector*> pvs;

    LevelBounds levelBounds;
    float targetDensity;
    
    std::unique_ptr<Field> spaceDecompositionField; /// a scalar field used to decompose the space with level sets

    int nSamples;                                   /// number of times we called sample function
    PinnedBuffer<unsigned long long int> nInsides;  /// number of samples per subregion
    std::vector<double> volumes;                    /// volume of each subregion

    std::vector<float> densities;
    PinnedBuffer<float> forces;
    
    std::vector<PidControl<float>> controllers;
    float Kp, Ki, Kd;

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
    ~PostprocessDensityControl();

    void deserialize(MPI_Status& stat) override;

private:
    FILE *fdump;
};
