#include "outlet.h"

#include <core/field/from_function.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/kernel_launch.h>

#include <memory>

using AccumulatedIntType = unsigned long long int;

namespace OutletPluginKernels
{

/// Erase all particles that satisfy `isInsideFunc(p.r)` with probability given
/// by `killProbabilityFunc()`. Value `killProbabilityFunc()` is a constant,
/// but in some cases it can only be computed on GPU, so it cannot be passed by
/// value.
template <typename IsInsideFunc, typename KillProbabilityFunc>
static __global__ void killParticles(PVview view, IsInsideFunc isInsideFunc, real seed, KillProbabilityFunc killProbabilityFunc)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= view.size) return;

    Particle p;
    view.readPosition(p, i);

    if (p.isMarked() || !isInsideFunc(p.r)) return;

    real prob = killProbabilityFunc();

    if (Saru::uniform01(seed, p.i1, i) >= prob) return;

    p.mark();
    // this will also write (wrong) velocity, but the particle will be killed by the cellLists
    // so we can avoid reading the velocity above
    view.writeParticle(i, p);
}

} // namespace OutletPluginKernels


OutletPlugin::OutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames) :
    SimulationPlugin(state, name),
    pvNames(pvNames)
{}

OutletPlugin::~OutletPlugin() = default;

void OutletPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pvs.reserve(pvNames.size());
    for (const auto& pvName : pvNames)
        pvs.push_back( simulation->getPVbyNameOrDie(pvName) );
}


PlaneOutletPlugin::PlaneOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames, real4 plane) :
    OutletPlugin(state, std::move(name), std::move(pvNames)),
    plane(plane)
{}

PlaneOutletPlugin::~PlaneOutletPlugin() = default;

void PlaneOutletPlugin::beforeCellLists(cudaStream_t stream)
{
    const int nthreads = 128;

    for (auto pv : pvs)
    {
        PVview view(pv, pv->local());

        real seed = udistr(gen);

        auto isInsideFunc = [plane = this->plane, domain = state->domain] __device__ (real3 r) {
            r = domain.local2global(r);
            return plane.x * r.x + plane.y * r.y + plane.z * r.z + plane.w >= 0.f;
        };
        auto killProbability = [] __device__ () { return 1.0f; };
        SAFE_KERNEL_LAUNCH(
            OutletPluginKernels::killParticles,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, isInsideFunc,seed, killProbability);
    }
}


namespace RegionOutletPluginKernels
{

static __device__ inline bool isInsideRegion(const FieldDeviceHandler& field, const real3& r)
{
    return field(r) < 0.f;
}

/// Monte-Carlo estimate of the region volume.
static __global__ void countInsideRegion(AccumulatedIntType nSamples, DomainInfo domain, FieldDeviceHandler field, real seed, AccumulatedIntType *nInside)
{
    AccumulatedIntType tid = threadIdx.x + blockIdx.x * blockDim.x;
    int countInside = 0;

    for (AccumulatedIntType i = tid; i < nSamples; i += blockDim.x * gridDim.x)
    {
        real3 r {Saru::uniform01(seed, i - 2, i + 4242),
                  Saru::uniform01(seed, i - 3, i + 4343),
                  Saru::uniform01(seed, i - 4, i + 4444)};

        r = domain.localSize * (r - 0.5f);

        if (isInsideRegion(field, r))
            ++countInside;
    }

    countInside = warpReduce(countInside, [](int a, int b) {return a+b;});

    if (laneId() == 0)
        atomicAdd(nInside, (AccumulatedIntType) countInside);
}

__global__ void countParticlesInside(PVview view, FieldDeviceHandler field, int *nInside)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int countInside = 0;

    if (i < view.size)
    {
        Particle p;
        view.readPosition(p, i);
        
        if (!p.isMarked() && isInsideRegion(field, p.r))
            ++countInside;
    }

    countInside = warpReduce(countInside, [](int a, int b) {return a+b;});

    if (laneId() == 0)
        atomicAdd(nInside, countInside);
}

} // namespace RegionOutletPluginKernels


RegionOutletPlugin::RegionOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                                       RegionFunc region, real3 resolution) :
    OutletPlugin(state, name, std::move(pvNames)),
    outletRegion(std::make_unique<FieldFromFunction>(state, name + "_region", region, resolution)),
    volume(0)
{}

RegionOutletPlugin::~RegionOutletPlugin() = default;

void RegionOutletPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    OutletPlugin::setup(simulation, comm, interComm);

    outletRegion->setup(comm);
    
    volume = computeVolume(1000000, udistr(gen));
}

double RegionOutletPlugin::computeVolume(long long int nSamples, real seed) const
{
    auto domain = state->domain;

    double totVolume = domain.localSize.x * domain.localSize.y * domain.localSize.z;

    PinnedBuffer<AccumulatedIntType> nInside(1);
    nInside.clearDevice(defaultStream);

    const int nthreads = 128;
    const int nblocks = math::min(getNblocks(nSamples, nthreads), 1024);
    
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


DensityOutletPlugin::DensityOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                                         real numberDensity, RegionFunc region, real3 resolution) :
    RegionOutletPlugin(state, std::move(name), std::move(pvNames), std::move(region), resolution),
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

        real seed = udistr(gen);
        real rhoTimesVolume = volume * numberDensity;

        auto isInsideFunc = [field = outletRegion->handler()] __device__ (const real3& r) {
            return RegionOutletPluginKernels::isInsideRegion(field, r);
        };
        auto killProbability = [rhoTimesVolume, nInside = nParticlesInside.devPtr()] __device__ () {
            int n = *nInside;
            return n > 0 ? (n - rhoTimesVolume) / n : 0.f;
        };
        SAFE_KERNEL_LAUNCH(
            OutletPluginKernels::killParticles,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, isInsideFunc, seed, killProbability);
    }
}



RateOutletPlugin::RateOutletPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                                   real rate, RegionFunc region, real3 resolution) :
    RegionOutletPlugin(state, std::move(name), std::move(pvNames), std::move(region), resolution),
    rate(rate)
{}

RateOutletPlugin::~RateOutletPlugin() = default;

void RateOutletPlugin::beforeCellLists(cudaStream_t stream)
{
    countInsideParticles(stream);

    const int  nthreads = 128;
    
    for (auto pv : pvs)
    {
        PVview view(pv, pv->local());

        real seed = udistr(gen);
        real QTimesdt = rate * state->dt * view.invMass;

        auto isInsideFunc = [field = outletRegion->handler()] __device__ (const real3& r) {
            return RegionOutletPluginKernels::isInsideRegion(field, r);
        };
        auto killProbability = [QTimesdt, nInside = nParticlesInside.devPtr()] __device__ () {
            int n = *nInside;
            return n > 0 ? QTimesdt / n : 0.f;
        };
        SAFE_KERNEL_LAUNCH(
            OutletPluginKernels::killParticles,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, isInsideFunc, seed, killProbability);
    }
}
