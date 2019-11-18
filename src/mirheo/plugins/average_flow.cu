#include "average_flow.h"

#include "utils/sampling_helpers.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace AverageFlowKernels
{

__global__ void sample(PVview pvView, CellListInfo cinfo,
                       real *avgDensity, ChannelsInfo channelsInfo)
{
    const int pid = threadIdx.x + blockIdx.x*blockDim.x;
    if (pid >= pvView.size) return;

    const Particle p(pvView.readParticle(pid));

    const int cid = cinfo.getCellId(p.r);

    atomicAdd(avgDensity + cid, 1.0_r);

    SamplingHelpersKernels::sampleChannels(pid, cid, channelsInfo);
}

} // namespace AverageFlowKernels

int Average3D::getNcomponents(Average3D::ChannelType type) const
{
    int components = 3;
    if (type == Average3D::ChannelType::Scalar)  components = 1;
    if (type == Average3D::ChannelType::Tensor6) components = 6;
    return components;
}

Average3D::Average3D(const MirState *state, std::string name,
                     std::vector<std::string> pvNames,
                     std::vector<std::string> channelNames,
                     int sampleEvery, int dumpEvery, real3 binSize) :
    SimulationPlugin(state, name),
    pvNames(pvNames),
    sampleEvery(sampleEvery),
    dumpEvery(dumpEvery),
    binSize(binSize)
{
    const size_t n = channelNames.size();
    
    channelsInfo.n = n;
    channelsInfo.types      .resize_anew(n);
    channelsInfo.averagePtrs.resize_anew(n);
    channelsInfo.dataPtrs   .resize_anew(n);
    channelsInfo.average    .resize     (n);
    accumulatedAverage      .resize     (n);

    channelsInfo.names = std::move(channelNames);
}

namespace average3DDetails
{
template <typename T>
static Average3D::ChannelType getChannelType(T) {return Average3D::ChannelType::None;}
static Average3D::ChannelType getChannelType(real )  {return Average3D::ChannelType::Scalar;}
static Average3D::ChannelType getChannelType(real3)  {return Average3D::ChannelType::Vector_real3;}
static Average3D::ChannelType getChannelType(real4)  {return Average3D::ChannelType::Vector_real4;}
static Average3D::ChannelType getChannelType(Force)  {return Average3D::ChannelType::Vector_real4;}
static Average3D::ChannelType getChannelType(Stress) {return Average3D::ChannelType::Tensor6;}
} // average3DDetails

static Average3D::ChannelType getChannelTypeFromChannelDesc(const std::string& name, const DataManager::ChannelDescription& desc)
{
    auto type = mpark::visit([](auto *pinnedBufferPtr)
    {
        using T = typename std::remove_pointer<decltype(pinnedBufferPtr)>::type::value_type;
        return average3DDetails::getChannelType(T());
    }, desc.varDataPtr);

    if (type == Average3D::ChannelType::None)
        die ("Channel '%s' is not supported for average flow plugin", name.c_str());

    return type;
}

void Average3D::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (const auto& pvName : pvNames)
        pvs.push_back(simulation->getPVbyNameOrDie(pvName));

    if (pvs.size() == 0)
        die("Plugin '%s' needs at least one particle vector", name.c_str());

    const LocalParticleVector *lpv = pvs[0]->local();
    
    // setup types from available channels
    for (int i = 0; i < channelsInfo.n; ++i)
    {
        const std::string& channelName = channelsInfo.names[i];
        const auto& desc = lpv->dataPerParticle.getChannelDescOrDie(channelName);
        
        const ChannelType type = getChannelTypeFromChannelDesc(channelName, desc);
        channelsInfo.types[i] = type;
    }
    
    rank3D   = simulation->rank3D;
    nranks3D = simulation->nranks3D;
    
    // TODO: this should be reworked if the domains are allowed to have different size
    resolution = make_int3( math::floor(state->domain.localSize / binSize) );
    binSize = state->domain.localSize / make_real3(resolution);

    if (resolution.x <= 0 || resolution.y <= 0 || resolution.z <= 0)
    	die("Plugin '%s' has to have at least 1 cell per rank per dimension, got %dx%dx%d."
            "Please decrease the bin size", resolution.x, resolution.y, resolution.z);

    const int total = resolution.x * resolution.y * resolution.z;

    numberDensity.resize_anew(total);
    numberDensity.clear(defaultStream);

    accumulatedNumberDensity.resize_anew(total);
    accumulatedNumberDensity.clear(defaultStream);
    
    std::string allChannels("density");

    for (int i = 0; i < channelsInfo.n; ++i)
    {
        const int components = getNcomponents(channelsInfo.types[i]);
        
        channelsInfo.average[i].resize_anew(components * total);
        accumulatedAverage  [i].resize_anew(components * total);
        
        channelsInfo.average[i].clear(defaultStream);
        accumulatedAverage  [i].clear(defaultStream);
        
        channelsInfo.averagePtrs[i] = channelsInfo.average[i].devPtr();

        allChannels += ", " + channelsInfo.names[i];
    }

    channelsInfo.averagePtrs .uploadToDevice(defaultStream);
    channelsInfo.types       .uploadToDevice(defaultStream);

    info("Plugin '%s' initialized for the %d PVs and channels %s, resolution %dx%dx%d",
         name.c_str(), pvs.size(), allChannels.c_str(),
         resolution.x, resolution.y, resolution.z);
}

void Average3D::sampleOnePv(ParticleVector *pv, cudaStream_t stream)
{
    CellListInfo cinfo(binSize, state->domain.localSize);
    PVview pvView(pv, pv->local());
    ChannelsInfo gpuInfo(channelsInfo, pv, stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH
        (AverageFlowKernels::sample,
         getNblocks(pvView.size, nthreads), nthreads, 0, stream,
         pvView, cinfo, numberDensity.devPtr(), gpuInfo);
}

static void accumulateOneArray(int n, int components, const real *src, double *dst, cudaStream_t stream)
{
    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH
        (SamplingHelpersKernels::accumulate,
         getNblocks(n * components, nthreads), nthreads, 0, stream,
         n, components, src, dst);
}

void Average3D::accumulateSampledAndClear(cudaStream_t stream)
{
    const int ncells = numberDensity.size();

    accumulateOneArray(ncells, 1, numberDensity.devPtr(), accumulatedNumberDensity.devPtr(), stream);
    numberDensity.clear(stream);    

    for (int i = 0; i < channelsInfo.n; ++i)
    {
        const int components = getNcomponents(channelsInfo.types[i]);

        accumulateOneArray
            (ncells, components,
             channelsInfo.average[i].devPtr(),
             accumulatedAverage  [i].devPtr(),
             stream);

        channelsInfo.average[i].clear(stream);
    }
}

void Average3D::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(state, sampleEvery)) return;

    debug2("Plugin %s is sampling now", name.c_str());

    for (auto& pv : pvs)
        sampleOnePv(pv, stream);

    accumulateSampledAndClear(stream);
    
    ++nSamples;
}

void Average3D::scaleSampled(cudaStream_t stream)
{
    constexpr int nthreads = 128;
    const int ncells = accumulatedNumberDensity.size();
    // Order is important here! First channels, only then density 

    for (int i = 0; i < channelsInfo.n; ++i)
    {
        auto& data = accumulatedAverage[i];

        const int components = getNcomponents(channelsInfo.types[i]);

        SAFE_KERNEL_LAUNCH
            (SamplingHelpersKernels::scaleVec,
             getNblocks(ncells, nthreads), nthreads, 0, stream,
             ncells, components, data.devPtr(), accumulatedNumberDensity.devPtr() );

        data.downloadFromDevice(stream, ContainersSynch::Asynch);
        data.clearDevice(stream);
    }

    SAFE_KERNEL_LAUNCH(
        SamplingHelpersKernels::scaleDensity,
        getNblocks(ncells, nthreads), nthreads, 0, stream,
        ncells, accumulatedNumberDensity.devPtr(), 1.0 / (nSamples * binSize.x*binSize.y*binSize.z) );

    accumulatedNumberDensity.downloadFromDevice(stream, ContainersSynch::Synch);
    accumulatedNumberDensity.clearDevice(stream);

    nSamples = 0;
}

void Average3D::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;
    if (nSamples == 0) return;
    
    scaleSampled(stream);

    const MirState::StepType timeStamp = getTimeStamp(state, dumpEvery) - 1;  // -1 to start from 0
    
    debug2("Plugin '%s' is now packing the data", name.c_str());
    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, state->currentTime, timeStamp, accumulatedNumberDensity, accumulatedAverage);
    send(sendBuffer);
}

void Average3D::handshake()
{
    std::vector<int> sizes;

    for (auto t : channelsInfo.types)
        sizes.push_back(getNcomponents(t));
    
    SimpleSerializer::serialize(sendBuffer, nranks3D, rank3D, resolution, binSize, sizes, channelsInfo.names);
    send(sendBuffer);
}

} // namespace mirheo
