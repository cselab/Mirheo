#include "average_relative_flow.h"

#include "utils/sampling_helpers.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>

namespace mirheo
{

namespace AverageRelativeFlowKernels
{
__global__ void sampleRelative(
        PVview pvView, CellListInfo cinfo,
        real* avgDensity,
        ChannelsInfo channelsInfo,
        real3 relativePoint)
{
    const int pid = threadIdx.x + blockIdx.x*blockDim.x;
    if (pid >= pvView.size) return;

    real3 r = make_real3(pvView.readPosition(pid));
    r -= relativePoint;

    int3 cid3 = cinfo.getCellIdAlongAxes<CellListsProjection::NoClamp>(r);
    cid3 = (cid3 + cinfo.ncells) % cinfo.ncells;
    const int cid = cinfo.encode(cid3);

    atomicAdd(avgDensity + cid, 1);

    SamplingHelpersKernels::sampleChannels(pid, cid, channelsInfo);
}
} // namespace AverageRelativeFlowKernels

AverageRelative3D::AverageRelative3D(
    const MirState *state, std::string name, std::vector<std::string> pvNames,
    std::vector<std::string> channelNames,
    std::vector<Average3D::ChannelType> channelTypes, int sampleEvery,
    int dumpEvery, real3 binSize, std::string relativeOVname, int relativeID) :
    Average3D(state, name, pvNames, channelNames, channelTypes, sampleEvery,
              dumpEvery, binSize),
    relativeOVname(relativeOVname), relativeID(relativeID)
{}

void AverageRelative3D::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    Average3D::setup(simulation, comm, interComm);

    int local_size = density.size();
    int global_size = local_size * nranks;
    
    localDensity.resize(local_size);
    density.resize_anew(global_size);
    accumulated_density.resize_anew(global_size);
    density.clear(0);

    localChannels.resize(channelsInfo.n);

    for (int i = 0; i < channelsInfo.n; i++) {
        local_size = channelsInfo.average[i].size();
        global_size = local_size * nranks;
        localChannels[i].resize(local_size);
        channelsInfo.average[i].resize_anew(global_size);
        accumulated_average [i].resize_anew(global_size);
        channelsInfo.average[i].clear(0);
        channelsInfo.averagePtrs[i] = channelsInfo.average[i].devPtr();
    }

    channelsInfo.averagePtrs.uploadToDevice(defaultStream);
    channelsInfo.types.uploadToDevice(defaultStream);

    // Relative stuff
    relativeOV = simulation->getOVbyNameOrDie(relativeOVname);

    if ( !relativeOV->local()->dataPerObject.checkChannelExists(ChannelNames::motions) )
        die("Only rigid objects are supported for relative flow, but got OV '%s'", relativeOV->name.c_str());

    int locsize = relativeOV->local()->nObjects;
    int totsize;

    MPI_Check( MPI_Reduce(&locsize, &totsize, 1, MPI_INT, MPI_SUM, 0, comm) );

    if (rank == 0 && relativeID >= totsize)
        die("Too few objects in OV '%s' (only %d); but requested id %d",
            relativeOV->name.c_str(), totsize, relativeID);
}

void AverageRelative3D::sampleOnePv(real3 relativeParam, ParticleVector *pv, cudaStream_t stream)
{
    CellListInfo cinfo(binSize, state->domain.globalSize);
    PVview pvView(pv, pv->local());
    ChannelsInfo gpuInfo(channelsInfo, pv, stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH
        (AverageRelativeFlowKernels::sampleRelative,
         getNblocks(pvView.size, nthreads), nthreads, 0, stream,
         pvView, cinfo, density.devPtr(), gpuInfo, relativeParam);
}

void AverageRelative3D::afterIntegration(cudaStream_t stream)
{
    const int TAG = 22;
    const int NCOMPONENTS = 2 * sizeof(real3) / sizeof(real);
    
    if (!isTimeEvery(state, sampleEvery)) return;

    debug2("Plugin %s is sampling now", name.c_str());

    real3 relativeParams[2] = {make_real3(0.0_r), make_real3(0.0_r)};

    // Find and broadcast the position and velocity of the relative object
    MPI_Request req;
    MPI_Check( MPI_Irecv(relativeParams, NCOMPONENTS, getMPIFloatType<real>(), MPI_ANY_SOURCE, TAG, comm, &req) );

    auto ids     = relativeOV->local()->dataPerObject.getData<int64_t>(ChannelNames::globalIds);
    auto motions = relativeOV->local()->dataPerObject.getData<RigidMotion>(ChannelNames::motions);

    ids    ->downloadFromDevice(stream, ContainersSynch::Asynch);
    motions->downloadFromDevice(stream, ContainersSynch::Synch);

    for (size_t i = 0; i < ids->size(); i++)
    {
        if ((*ids)[i] == relativeID)
        {
            real3 params[2] = { make_real3( (*motions)[i].r   ),
                                 make_real3( (*motions)[i].vel ) };

            params[0] = state->domain.local2global(params[0]);

            for (int r = 0; r < nranks; r++)
                MPI_Send(&params, NCOMPONENTS, getMPIFloatType<real>(), r, TAG, comm);

            break;
        }
    }

    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );

    relativeParams[0] = state->domain.global2local(relativeParams[0]);

    for (auto& pv : pvs) sampleOnePv(relativeParams[0], pv, stream);

    accumulateSampledAndClear(stream);
    
    averageRelativeVelocity += relativeParams[1];

    nSamples++;
}


void AverageRelative3D::extractLocalBlock()
{
    static const double scale_by_density = -1.0;
    
    auto oneChannel = [this] (const PinnedBuffer<double>& channel, Average3D::ChannelType type, double scale, std::vector<double>& dest) {

        MPI_Check( MPI_Allreduce(MPI_IN_PLACE, channel.hostPtr(), channel.size(), MPI_DOUBLE, MPI_SUM, comm) );

        int ncomponents = this->getNcomponents(type);

        int3 globalResolution = resolution * nranks3D;

        double factor;
        int dstId = 0;
        for (int k = rank3D.z*resolution.z; k < (rank3D.z+1)*resolution.z; k++) {
            for (int j = rank3D.y*resolution.y; j < (rank3D.y+1)*resolution.y; j++) {
                for (int i = rank3D.x*resolution.x; i < (rank3D.x+1)*resolution.x; i++) {                    
                    int scalId = (k*globalResolution.y*globalResolution.x + j*globalResolution.x + i);
                    int srcId = ncomponents * scalId;
                    for (int c = 0; c < ncomponents; c++) {
                        if (scale == scale_by_density) factor = 1.0_r / accumulated_density[scalId];
                        else                           factor = scale;

                        dest[dstId++] = channel[srcId] * factor;
                        srcId++;
                    }
                }
            }
        }
    };

    // Order is important! Density comes first
    oneChannel(accumulated_density, Average3D::ChannelType::Scalar, 1.0 / (nSamples * binSize.x*binSize.y*binSize.z), localDensity);

    for (int i = 0; i < channelsInfo.n; i++)
        oneChannel(accumulated_average[i], channelsInfo.types[i], scale_by_density, localChannels[i]);
}

void AverageRelative3D::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    for (int i = 0; i < channelsInfo.n; i++) {
        auto& data = accumulated_average[i];

        if (channelsInfo.names[i] == "velocity") {
            const int nthreads = 128;

            SAFE_KERNEL_LAUNCH
                (SamplingHelpersKernels::correctVelocity,
                 getNblocks(data.size() / 3, nthreads), nthreads, 0, stream,
                 data.size() / 3, (double3*)data.devPtr(), accumulated_density.devPtr(), averageRelativeVelocity / (real) nSamples);

            averageRelativeVelocity = make_real3(0);
        }
    }

        
    accumulated_density.downloadFromDevice(stream, ContainersSynch::Asynch);
    accumulated_density.clearDevice(stream);
    
    for (auto& data : accumulated_average)
    {
        data.downloadFromDevice(stream, ContainersSynch::Asynch);
        data.clearDevice(stream);
    }

    CUDA_Check( cudaStreamSynchronize(stream) );

    extractLocalBlock();
    nSamples = 0;

    MirState::StepType timeStamp = getTimeStamp(state, dumpEvery) - 1; // -1 to start from 0

    debug2("Plugin '%s' is now packing the data", name.c_str());
    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, state->currentTime, timeStamp, localDensity, localChannels);
    send(sendBuffer);
}

} // namespace mirheo
