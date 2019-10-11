#pragma once

#include "interface.h"

#include <core/containers.h>

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

class ParticleVector;
class CellList;
class Field;

class OutletPlugin : public SimulationPlugin
{
public:
    OutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames);
    ~OutletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override { return false; }

protected:

    std::vector<std::string> pvNames;
    std::vector<ParticleVector*> pvs;

    DeviceBuffer<int> nParticlesInside {1};

    std::mt19937 gen {42};
    std::uniform_real_distribution<float> udistr {0.f, 1.f};
};


class PlaneOutletPlugin : public OutletPlugin
{
public:
    PlaneOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvName, float4 plane);

    ~PlaneOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;

private:
    float4 plane;
};


class RegionOutletPlugin : public OutletPlugin
{
public:
    using RegionFunc = std::function<float(float3)>;
    
    RegionOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                       RegionFunc region, float3 resolution);

    ~RegionOutletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override { return false; }

protected:

    double computeVolume(long long int nSamples, float seed) const;
    void countInsideParticles(cudaStream_t stream);
    
protected:
    
    double volume;

    std::unique_ptr<Field> outletRegion;

    DeviceBuffer<int> nParticlesInside {1};

    std::mt19937 gen {42};
    std::uniform_real_distribution<float> udistr {0.f, 1.f};
};


class DensityOutletPlugin : public RegionOutletPlugin
{
public:
    
    DensityOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                        float numberDensity, RegionFunc region, float3 resolution);

    ~DensityOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;
    
protected:
    
    float numberDensity;
};


class RateOutletPlugin : public RegionOutletPlugin
{
public:
    
    RateOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                     float rate, RegionFunc region, float3 resolution);

    ~RateOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;
    
protected:
    
    float rate;
};
