#include "average_flow.h"

#include <core/utils/kernel_launch.h>
#include <core/simulation.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>

#include "simple_serializer.h"
#include "sampling_helpers.h"

__global__ void sample(
        PVview pvView, CellListInfo cinfo,
        float* avgDensity,
        ChannelsInfo channelsInfo)
{
    const int pid = threadIdx.x + blockIdx.x*blockDim.x;
    if (pid >= pvView.size) return;

    Particle p(pvView.particles, pid);

    int cid = cinfo.getCellId(p.r);

    atomicAdd(avgDensity + cid, 1);

    sampleChannels(pid, cid, channelsInfo);
}

Average3D::Average3D(std::string name,
        std::string pvName,
        std::vector<std::string> channelNames, std::vector<Average3D::ChannelType> channelTypes,
        int sampleEvery, int dumpEvery, float3 binSize) :
    SimulationPlugin(name), pvName(pvName),
    sampleEvery(sampleEvery), dumpEvery(dumpEvery), binSize(binSize),
    nSamples(0)
{
    channelsInfo.n = channelTypes.size();
    channelsInfo.types.resize_anew(channelsInfo.n);
    channelsInfo.average.resize(channelsInfo.n);
    channelsInfo.averagePtrs.resize_anew(channelsInfo.n);
    channelsInfo.dataPtrs.resize_anew(channelsInfo.n);

    for (int i=0; i<channelsInfo.n; i++)
        channelsInfo.types[i] = channelTypes[i];

    channelsInfo.names = channelNames;
}

void Average3D::setup(Simulation* sim, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(sim, comm, interComm);

    // TODO: this should be reworked if the domains are allowed to have different size
    resolution = make_int3( floorf(sim->domain.localSize / binSize) );
    binSize = sim->domain.localSize / make_float3(resolution);

    const int total = resolution.x * resolution.y * resolution.z;

    density.resize_anew(total);
    density.clear(0);
    std::string allChannels("density");
    for (int i=0; i<channelsInfo.n; i++)
    {
        if      (channelsInfo.types[i] == Average3D::ChannelType::Scalar)  channelsInfo.average[i].resize_anew(1*total);
        else if (channelsInfo.types[i] == Average3D::ChannelType::Tensor6) channelsInfo.average[i].resize_anew(6*total);
        else                                                               channelsInfo.average[i].resize_anew(3*total);

        channelsInfo.average[i].clear(0);
        channelsInfo.averagePtrs[i] = channelsInfo.average[i].devPtr();

        allChannels += " ," + channelsInfo.names[i];
    }

    channelsInfo.averagePtrs.uploadToDevice(0);

    pv = sim->getPVbyNameOrDie(pvName);

    info("Plugin %s initialized for the PV '%s' and channels %s, resolution %dx%dx%d",
            name.c_str(), pv->name.c_str(), allChannels.c_str(),
            resolution.x, resolution.y, resolution.z);
}



void Average3D::afterIntegration(cudaStream_t stream)
{
    if (currentTimeStep % sampleEvery != 0 || currentTimeStep == 0) return;

    debug2("Plugin %s is sampling now", name.c_str());

    CellListInfo cinfo(binSize, pv->domain.localSize);
    PVview pvView(pv, pv->local());
    ChannelsInfo gpuInfo(channelsInfo, pv, stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            sample,
            getNblocks(pvView.size, nthreads), nthreads, 0, stream,
            pvView, cinfo, density.devPtr(), gpuInfo);

    nSamples++;
}

void Average3D::scaleSampled(cudaStream_t stream)
{
    const int nthreads = 128;
    // Order is important here! First channels, only then dens

    for (int i=0; i<channelsInfo.n; i++)
    {
        auto& data = channelsInfo.average[i];
        int sz = density.size();
        int components = 3;
        if (channelsInfo.types[i] == ChannelType::Scalar)  components = 1;
        if (channelsInfo.types[i] == ChannelType::Tensor6) components = 6;

        SAFE_KERNEL_LAUNCH(
                scaleVec,
                getNblocks(sz, nthreads), nthreads, 0, stream,
                sz, components, data.devPtr(), density.devPtr() );

        data.downloadFromDevice(stream, ContainersSynch::Asynch);
        data.clearDevice(stream);
    }

    int sz = density.size();
    SAFE_KERNEL_LAUNCH(
            scaleDensity,
            getNblocks(sz, nthreads), nthreads, 0, stream,
            sz, density.devPtr(), pv->mass / (nSamples * binSize.x*binSize.y*binSize.z) );

    density.downloadFromDevice(stream, ContainersSynch::Synch);
    density.clearDevice(stream);

    nSamples = 0;
}

void Average3D::serializeAndSend(cudaStream_t stream)
{
    if (currentTimeStep % dumpEvery != 0 || currentTimeStep == 0) return;
    if (nSamples == 0) return;
    
    scaleSampled(stream);

    // Calculate total size for sending
    int totalSize = SimpleSerializer::totSize(currentTime, density);
    for (auto& ch : channelsInfo.average)
        totalSize += SimpleSerializer::totSize(ch);

    // Now allocate the sending buffer and pack everything into it
    debug2("Plugin %s is packing now data", name.c_str());
    sendBuffer.resize(totalSize);
    SimpleSerializer::serialize(sendBuffer.data(), currentTime, density);
    int currentSize = SimpleSerializer::totSize(currentTime, density);

    for (int i=0; i<channelsInfo.n; i++)
    {
        SimpleSerializer::serialize(sendBuffer.data() + currentSize, channelsInfo.average[i]);
        currentSize += SimpleSerializer::totSize(channelsInfo.average[i]);
    }

    send(sendBuffer);
}

void Average3D::handshake()
{
    std::vector<char> data;
    std::vector<int> sizes;

    // for density
    sizes.push_back(1);

    for (auto t : channelsInfo.types)
        switch (t)
        {
            case ChannelType::Scalar:
                sizes.push_back(1);
                break;
            case ChannelType::Tensor6:
                sizes.push_back(6);
                break;
            default:
                sizes.push_back(3);
                break;
        }

    std::string dens("density");
    SimpleSerializer::serialize(data, sim->nranks3D, sim->rank3D, resolution, binSize, sizes, dens);

    int namesSize = 0;
    for (auto& s : channelsInfo.names)
        namesSize += SimpleSerializer::totSize(s);

    int shift = data.size();
    data.resize(data.size() + namesSize);

    for (auto& s : channelsInfo.names)
    {
        SimpleSerializer::serialize(data.data() + shift, s);
        shift += SimpleSerializer::totSize(s);
    }

    send(data.data(), data.size());
}

