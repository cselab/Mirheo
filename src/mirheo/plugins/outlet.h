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
class Field;

class OutletPlugin : public SimulationPlugin
{
public:
    OutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames);
    ~OutletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override { return false; }

protected:

    std::vector<std::string> pvNames_;
    std::vector<ParticleVector*> pvs_;

    DeviceBuffer<int> nParticlesInside_ {1};

    std::mt19937 gen_ {42};
    std::uniform_real_distribution<real> udistr_ {0._r, 1._r};
};


class PlaneOutletPlugin : public OutletPlugin
{
public:
    PlaneOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvName, real4 plane);

    ~PlaneOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;

private:
    real4 plane_;
};


class RegionOutletPlugin : public OutletPlugin
{
public:
    using RegionFunc = std::function<real(real3)>;

    RegionOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                       RegionFunc region, real3 resolution);

    ~RegionOutletPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override { return false; }

protected:

    double computeVolume(long long int nSamples, real seed) const;
    void countInsideParticles(cudaStream_t stream);

protected:

    double volume_;
    std::unique_ptr<Field> outletRegion_;
};


class DensityOutletPlugin : public RegionOutletPlugin
{
public:

    DensityOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                        real numberDensity, RegionFunc region, real3 resolution);

    ~DensityOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;

protected:

    real numberDensity_;
};


class RateOutletPlugin : public RegionOutletPlugin
{
public:

    RateOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                     real rate, RegionFunc region, real3 resolution);

    ~RateOutletPlugin();

    void beforeCellLists(cudaStream_t stream) override;

protected:

    real rate_;
};

} // namespace mirheo
