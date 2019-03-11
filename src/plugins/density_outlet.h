#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/field/interface.h>

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

class ParticleVector;
class CellList;

class DensityOutletPlugin : public SimulationPlugin
{
public:

    using RegionFunc = std::function<float(float3)>;
    
    DensityOutletPlugin(const YmrState *state, std::string name, std::vector<std::string> pvNames,
                        float numberDensity, RegionFunc region, float3 resolution);

    ~DensityOutletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:

    double computeVolume(long long int nSamples, float seed) const;
    
private:
    
    std::vector<std::string> pvNames;
    std::vector<ParticleVector*> pvs;
    
    float numberDensity;
    float volume;

    std::unique_ptr<Field> outletRegion;

    DeviceBuffer<int> nParticlesInside {1};

    std::mt19937 gen {42};
    std::uniform_real_distribution<float> udistr {0.f, 1.f};
};
