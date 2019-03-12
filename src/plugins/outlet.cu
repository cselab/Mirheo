#include "outlet.h"

#include <core/field/from_function.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/make_unique.h>

using AccumulatedIntType = unsigned long long int;

namespace RegionOutletPluginKernels
{

__device__ inline bool isInsideRegion(const FieldDeviceHandler& field, const float3& r)
{
    return field(r) < 0.f;
}

__global__ void countInsideRegion(AccumulatedIntType nSamples, DomainInfo domain, FieldDeviceHandler field, float seed, AccumulatedIntType *nInside)
{
    AccumulatedIntType tid = threadIdx.x + blockIdx.x * blockDim.x;
    int countInside = 0;

    for (AccumulatedIntType i = tid; i < nSamples; i += blockDim.x * gridDim.x)
    {
        float3 r {Saru::uniform01(seed, i - 2, i + 4242),
                  Saru::uniform01(seed, i - 3, i + 4343),
                  Saru::uniform01(seed, i - 4, i + 4444)};

        r = domain.localSize * (r - 0.5f);

        if (isInsideRegion(field, r))
            ++countInside;
    }

    countInside = warpReduce(countInside, [](int a, int b) {return a+b;});

    if (__laneid() == 0)
        atomicAdd(nInside, (AccumulatedIntType) countInside);
}

__global__ void countParticlesInside(PVview view, FieldDeviceHandler field, int *nInside)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int countInside = 0;

    if (i < view.size)
    {
        Particle p;
        p.readCoordinate(view.particles, i);
        
        if (!p.isMarked() && isInsideRegion(field, p.r))
            ++countInside;
    }

    countInside = warpReduce(countInside, [](int a, int b) {return a+b;});

    if (__laneid() == 0)
        atomicAdd(nInside, countInside);
}

} // namespace RegionOutletPluginKernels


RegionOutletPlugin::RegionOutletPlugin(const YmrState *state, std::string name, std::vector<std::string> pvNames,
                                       RegionFunc region, float3 resolution) :
    SimulationPlugin(state, name),
    pvNames(pvNames),
    outletRegion(std::make_unique<FieldFromFunction>(state, name + "_region", region, resolution)),
    volume(0)
{}

RegionOutletPlugin::~RegionOutletPlugin() = default;

void RegionOutletPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (const auto& pvName : pvNames)
        pvs.push_back( simulation->getPVbyNameOrDie(pvName) );

    outletRegion->setup(comm);
    
    volume = computeVolume(1000000, udistr(gen));
}

double RegionOutletPlugin::computeVolume(long long int nSamples, float seed) const
{
    auto domain = state->domain;

    double totVolume = domain.localSize.x * domain.localSize.y * domain.localSize.z;

    PinnedBuffer<AccumulatedIntType> nInside(1);
    nInside.clearDevice(defaultStream);

    const int nthreads = 128;
    const int nblocks = std::min(getNblocks(nSamples, nthreads), 1024);
    
    SAFE_KERNEL_LAUNCH(
        RegionOutletPluginKernels::countInsideRegion,
        nblocks, nthreads, 0, defaultStream,
        nSamples, domain, outletRegion->handler(),
        seed, nInside.devPtr());

    nInside.downloadFromDevice(defaultStream);

    return totVolume * (double) nInside[0] / (double) nSamples;
}

void RegionOutletPlugin::countInsideParticles(cudaStream_t stream)
{
    nParticlesInside.clearDevice(stream);

    const int nthreads = 128;
    
    for (auto pv : pvs)
    {
        PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
            RegionOutletPluginKernels::countParticlesInside,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, outletRegion->handler(), nParticlesInside.devPtr());        
    }    
}


namespace DensityOutletPluginKernels
{
using namespace RegionOutletPluginKernels;

__global__ void killParticles(PVview view, FieldDeviceHandler field, const int *nInside, float seed, float rhoTimesVolume)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= view.size) return;

    Particle p;
    p.readCoordinate(view.particles, i);
        
    if (p.isMarked() || !isInsideRegion(field, p.r)) return;

    int n = *nInside;

    float prob = (n - rhoTimesVolume) / n;

    if (Saru::uniform01(seed, p.i1, i) >= prob) return;

    p.mark();
    // this will also write (wrong) velocity, but the particle will be killed by the cellLists
    // so we can avoid reading the velocity above
    p.write2Float4(view.particles, i); 
}

} // namespace DensityOutletPluginKernels

DensityOutletPlugin::DensityOutletPlugin(const YmrState *state, std::string name, std::vector<std::string> pvNames,
                                         float numberDensity, RegionFunc region, float3 resolution) :
    RegionOutletPlugin(state, name, pvNames, region, resolution),
    numberDensity(numberDensity)
{}

DensityOutletPlugin::~DensityOutletPlugin() = default;

void DensityOutletPlugin::beforeCellLists(cudaStream_t stream)
{
    countInsideParticles(stream);

    const int nthreads = 128;
    
    for (auto pv : pvs)
    {
        PVview view(pv, pv->local());

        float seed = udistr(gen);
        float rhoTimesVolume = volume * numberDensity;

        SAFE_KERNEL_LAUNCH(
            DensityOutletPluginKernels::killParticles,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, outletRegion->handler(), nParticlesInside.devPtr(), seed, rhoTimesVolume);        
    }    
}

