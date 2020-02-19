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
    pvNames_(pvNames),
    sampleEvery_(sampleEvery),
    dumpEvery_(dumpEvery),
    binSize_(binSize)
{
    const size_t n = channelNames.size();
    
    channelsInfo_.n = n;
    channelsInfo_.types      .resize_anew(n);
    channelsInfo_.averagePtrs.resize_anew(n);
    channelsInfo_.dataPtrs   .resize_anew(n);
    channelsInfo_.average    .resize     (n);
    accumulatedAverage_      .resize     (n);

    channelsInfo_.names = std::move(channelNames);
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

    for (const auto& pvName : pvNames_)
        pvs_.push_back(simulation->getPVbyNameOrDie(pvName));

    if (pvs_.size() == 0)
        die("Plugin '%s' needs at least one particle vector", getCName());

    const LocalParticleVector *lpv = pvs_[0]->local();
    
    // setup types from available channels
    for (int i = 0; i < channelsInfo_.n; ++i)
    {
        const std::string& channelName = channelsInfo_.names[i];
        const auto& desc = lpv->dataPerParticle.getChannelDescOrDie(channelName);
        
        const ChannelType type = getChannelTypeFromChannelDesc(channelName, desc);
        channelsInfo_.types[i] = type;
    }
    
    rank3D_   = simulation->getRank3D();
    nranks3D_ = simulation->getNRanks3D();
    
    // TODO: this should be reworked if the domains are allowed to have different size
    resolution_ = make_int3( math::floor(getState()->domain.localSize / binSize_) );
    binSize_ = getState()->domain.localSize / make_real3(resolution_);

    if (resolution_.x <= 0 || resolution_.y <= 0 || resolution_.z <= 0) {
    	die("Plugin '%s' has to have at least 1 cell per rank per dimension, got %dx%dx%d."
            "Please decrease the bin size",
            getCName(), resolution_.x, resolution_.y, resolution_.z);
    }

    const int total = resolution_.x * resolution_.y * resolution_.z;

    numberDensity_.resize_anew(total);
    numberDensity_.clear(defaultStream);

    accumulatedNumberDensity_.resize_anew(total);
    accumulatedNumberDensity_.clear(defaultStream);
    
    std::string allChannels = numberDensityChannelName_;

    for (int i = 0; i < channelsInfo_.n; ++i)
    {
        const int components = getNcomponents(channelsInfo_.types[i]);
        
        channelsInfo_.average[i].resize_anew(components * total);
        accumulatedAverage_  [i].resize_anew(components * total);
        
        channelsInfo_.average[i].clear(defaultStream);
        accumulatedAverage_  [i].clear(defaultStream);
        
        channelsInfo_.averagePtrs[i] = channelsInfo_.average[i].devPtr();

        allChannels += ", " + channelsInfo_.names[i];
    }

    channelsInfo_.averagePtrs .uploadToDevice(defaultStream);
    channelsInfo_.types       .uploadToDevice(defaultStream);

    info("Plugin '%s' initialized for the %zu PVs and channels %s, resolution %dx%dx%d",
         getCName(), pvs_.size(), allChannels.c_str(),
         resolution_.x, resolution_.y, resolution_.z);
}

void Average3D::sampleOnePv(ParticleVector *pv, cudaStream_t stream)
{
    CellListInfo cinfo(binSize_, getState()->domain.localSize);
    PVview pvView(pv, pv->local());
    ChannelsInfo gpuInfo(channelsInfo_, pv, stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH
        (AverageFlowKernels::sample,
         getNblocks(pvView.size, nthreads), nthreads, 0, stream,
         pvView, cinfo, numberDensity_.devPtr(), gpuInfo);
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
    const int ncells = numberDensity_.size();

    accumulateOneArray(ncells, 1, numberDensity_.devPtr(), accumulatedNumberDensity_.devPtr(), stream);
    numberDensity_.clear(stream);    

    for (int i = 0; i < channelsInfo_.n; ++i)
    {
        const int components = getNcomponents(channelsInfo_.types[i]);

        accumulateOneArray
            (ncells, components,
             channelsInfo_.average[i].devPtr(),
             accumulatedAverage_  [i].devPtr(),
             stream);

        channelsInfo_.average[i].clear(stream);
    }
}

void Average3D::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), sampleEvery_)) return;

    debug2("Plugin %s is sampling now", getCName());

    for (auto& pv : pvs_)
        sampleOnePv(pv, stream);

    accumulateSampledAndClear(stream);
    
    ++nSamples_;
}

void Average3D::scaleSampled(cudaStream_t stream)
{
    constexpr int nthreads = 128;
    const int ncells = accumulatedNumberDensity_.size();
    // Order is important here! First channels, only then density 

    for (int i = 0; i < channelsInfo_.n; ++i)
    {
        auto& data = accumulatedAverage_[i];

        const int components = getNcomponents(channelsInfo_.types[i]);

        SAFE_KERNEL_LAUNCH
            (SamplingHelpersKernels::scaleVec,
             getNblocks(ncells, nthreads), nthreads, 0, stream,
             ncells, components, data.devPtr(), accumulatedNumberDensity_.devPtr() );

        data.downloadFromDevice(stream, ContainersSynch::Asynch);
        data.clearDevice(stream);
    }

    SAFE_KERNEL_LAUNCH(
        SamplingHelpersKernels::scaleDensity,
        getNblocks(ncells, nthreads), nthreads, 0, stream,
        ncells, accumulatedNumberDensity_.devPtr(), 1.0 / (nSamples_ * binSize_.x*binSize_.y*binSize_.z) );

    accumulatedNumberDensity_.downloadFromDevice(stream, ContainersSynch::Synch);
    accumulatedNumberDensity_.clearDevice(stream);

    nSamples_ = 0;
}

void Average3D::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;
    if (nSamples_ == 0) return;
    
    scaleSampled(stream);

    const MirState::StepType timeStamp = getTimeStamp(getState(), dumpEvery_) - 1;  // -1 to start from 0
    
    debug2("Plugin '%s' is now packing the data", getCName());
    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, getState()->currentTime, timeStamp, accumulatedNumberDensity_, accumulatedAverage_);
    send(sendBuffer_);
}

void Average3D::handshake()
{
    std::vector<int> sizes;

    for (auto t : channelsInfo_.types)
        sizes.push_back(getNcomponents(t));
    
    SimpleSerializer::serialize(sendBuffer_, nranks3D_, rank3D_, resolution_, binSize_, sizes, channelsInfo_.names, numberDensityChannelName_);
    send(sendBuffer_);
}

const std::string Average3D::numberDensityChannelName_ = "number_densities";

} // namespace mirheo
