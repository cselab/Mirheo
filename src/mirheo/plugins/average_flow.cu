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

namespace AverageFlowKernels
{

__global__ void sample(
        PVview pvView, CellListInfo cinfo,
        real* avgDensity,
        ChannelsInfo channelsInfo)
{
    const int pid = threadIdx.x + blockIdx.x*blockDim.x;
    if (pid >= pvView.size) return;

    Particle p(pvView.readParticle(pid));

    int cid = cinfo.getCellId(p.r);

    atomicAdd(avgDensity + cid, 1);

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
                     std::vector<std::string> channelNames, std::vector<Average3D::ChannelType> channelTypes,
                     int sampleEvery, int dumpEvery, real3 binSize) :
    SimulationPlugin(state, name), pvNames(pvNames),
    sampleEvery(sampleEvery), dumpEvery(dumpEvery), binSize(binSize),
    nSamples(0)
{
    channelsInfo.n = channelTypes.size();
    channelsInfo.types.resize_anew(channelsInfo.n);
    channelsInfo.average.resize(channelsInfo.n);
    channelsInfo.averagePtrs.resize_anew(channelsInfo.n);
    channelsInfo.dataPtrs.resize_anew(channelsInfo.n);

    accumulated_average.resize(channelsInfo.n);

    for (int i=0; i<channelsInfo.n; i++)
        channelsInfo.types[i] = channelTypes[i];

    channelsInfo.names = channelNames;
}

void Average3D::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    rank3D   = simulation->rank3D;
    nranks3D = simulation->nranks3D;
    
    // TODO: this should be reworked if the domains are allowed to have different size
    resolution = make_int3( math::floor(state->domain.localSize / binSize) );
    binSize = state->domain.localSize / make_real3(resolution);

    if (resolution.x <= 0 || resolution.y <= 0 || resolution.z <= 0)
    	die("Plugin '%s' has to have at least 1 cell per rank per dimension, got %dx%dx%d."
            "Please decrease the bin size", resolution.x, resolution.y, resolution.z);

    const int total = resolution.x * resolution.y * resolution.z;

    density.resize_anew(total);
    density.clear(defaultStream);

    accumulated_density.resize_anew(total);
    accumulated_density.clear(0);
    
    std::string allChannels("density");

    for (int i = 0; i < channelsInfo.n; i++) {
        int components = getNcomponents(channelsInfo.types[i]);
        
        channelsInfo.average[i].resize_anew(components * total);
        accumulated_average [i].resize_anew(components * total);
        
        channelsInfo.average[i].clear(defaultStream);
        accumulated_average [i].clear(defaultStream);
        
        channelsInfo.averagePtrs[i] = channelsInfo.average[i].devPtr();

        allChannels += ", " + channelsInfo.names[i];
    }

    channelsInfo.averagePtrs.uploadToDevice(defaultStream);
    channelsInfo.types.uploadToDevice(defaultStream);

    for (const auto& pvName : pvNames)
        pvs.push_back(simulation->getPVbyNameOrDie(pvName));

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
         pvView, cinfo, density.devPtr(), gpuInfo);
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
    const int ncells = density.size();

    accumulateOneArray(ncells, 1, density.devPtr(), accumulated_density.devPtr(), stream);
    density.clear(stream);    

    for (int i = 0; i < channelsInfo.n; i++) {

        int components = getNcomponents(channelsInfo.types[i]);

        accumulateOneArray
            (ncells, components,
             channelsInfo.average[i].devPtr(),
             accumulated_average [i].devPtr(),
             stream);

        channelsInfo.average[i].clear(stream);
    }
}

void Average3D::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(state, sampleEvery)) return;

    debug2("Plugin %s is sampling now", name.c_str());

    for (auto& pv : pvs) sampleOnePv(pv, stream);

    accumulateSampledAndClear(stream);
    
    nSamples++;
}

void Average3D::scaleSampled(cudaStream_t stream)
{
    const int nthreads = 128;
    const int ncells = accumulated_density.size();
    // Order is important here! First channels, only then dens

    for (int i = 0; i < channelsInfo.n; i++) {
        auto& data = accumulated_average[i];

        int components = getNcomponents(channelsInfo.types[i]);

        SAFE_KERNEL_LAUNCH
            (SamplingHelpersKernels::scaleVec,
             getNblocks(ncells, nthreads), nthreads, 0, stream,
             ncells, components, data.devPtr(), accumulated_density.devPtr() );

        data.downloadFromDevice(stream, ContainersSynch::Asynch);
        data.clearDevice(stream);
    }

    SAFE_KERNEL_LAUNCH(
            SamplingHelpersKernels::scaleDensity,
            getNblocks(ncells, nthreads), nthreads, 0, stream,
            ncells, accumulated_density.devPtr(), 1.0 / (nSamples * binSize.x*binSize.y*binSize.z) );

    accumulated_density.downloadFromDevice(stream, ContainersSynch::Synch);
    accumulated_density.clearDevice(stream);

    nSamples = 0;
}

void Average3D::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;
    if (nSamples == 0) return;
    
    scaleSampled(stream);

    MirState::StepType timeStamp = getTimeStamp(state, dumpEvery) - 1;  // -1 to start from 0
    
    debug2("Plugin '%s' is now packing the data", name.c_str());
    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, state->currentTime, timeStamp, accumulated_density, accumulated_average);
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

