// Copyright 2020 ETH Zurich. All Rights Reserved.
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

namespace average_relative_flow_kernels
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

    sampling_helpers_kernels::sampleChannels(pid, cid, channelsInfo);
}
} // namespace average_relative_flow_kernels

AverageRelative3D::AverageRelative3D(
       const MirState *state, std::string name, std::vector<std::string> pvNames,
       std::vector<std::string> channelNames, int sampleEvery,
       int dumpEvery, real3 binSize, std::string relativeOVname, int relativeID) :
    Average3D(state, name, pvNames, channelNames, sampleEvery, dumpEvery, binSize),
    relativeOVname_(relativeOVname),
    relativeID_(relativeID)
{}

void AverageRelative3D::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    Average3D::setup(simulation, comm, interComm);

    int local_size = numberDensity_.size();
    int global_size = local_size * nranks_;

    localNumberDensity_      .resize(local_size);
    numberDensity_           .resize_anew(global_size);
    accumulatedNumberDensity_.resize_anew(global_size);
    numberDensity_.clear(defaultStream);

    localChannels_.resize(channelsInfo_.n);

    for (int i = 0; i < channelsInfo_.n; ++i)
    {
        local_size  = channelsInfo_.average[i].size();
        global_size = local_size * nranks_;
        localChannels_[i].resize(local_size);
        channelsInfo_.average[i].resize_anew(global_size);
        accumulatedAverage_  [i].resize_anew(global_size);
        channelsInfo_.average[i].clear(defaultStream);
        channelsInfo_.averagePtrs[i] = channelsInfo_.average[i].devPtr();
    }

    channelsInfo_.averagePtrs.uploadToDevice(defaultStream);
    channelsInfo_.types.uploadToDevice(defaultStream);

    // Relative stuff
    relativeOV_ = simulation->getOVbyNameOrDie(relativeOVname_);

    if ( !relativeOV_->local()->dataPerObject.checkChannelExists(channel_names::motions) )
        die("Only rigid objects are supported for relative flow, but got OV '%s'", relativeOV_->getCName());

    const int locsize = relativeOV_->local()->getNumObjects();
    int totsize {0};

    MPI_Check( MPI_Reduce(&locsize, &totsize, 1, MPI_INT, MPI_SUM, 0, comm) );

    if (rank_ == 0 && relativeID_ >= totsize)
        die("Too few objects in OV '%s' (only %d); but requested id %d",
            relativeOV_->getCName(), totsize, relativeID_);
}

void AverageRelative3D::sampleOnePv(real3 relativeParam, ParticleVector *pv, cudaStream_t stream)
{
    const CellListInfo cinfo(binSize_, getState()->domain.globalSize);
    PVview pvView(pv, pv->local());
    ChannelsInfo gpuInfo(channelsInfo_, pv, stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH
        (average_relative_flow_kernels::sampleRelative,
         getNblocks(pvView.size, nthreads), nthreads, 0, stream,
         pvView, cinfo, numberDensity_.devPtr(), gpuInfo, relativeParam);
}

void AverageRelative3D::afterIntegration(cudaStream_t stream)
{
    const int TAG = 22;
    const int NCOMPONENTS = 2 * sizeof(real3) / sizeof(real);

    if (!isTimeEvery(getState(), sampleEvery_)) return;

    debug2("Plugin %s is sampling now", getCName());

    real3 relativeParams[2] = {make_real3(0.0_r), make_real3(0.0_r)};

    // Find and broadcast the position and velocity of the relative object
    MPI_Request req;
    MPI_Check( MPI_Irecv(relativeParams, NCOMPONENTS, getMPIFloatType<real>(), MPI_ANY_SOURCE, TAG, comm_, &req) );

    auto ids     = relativeOV_->local()->dataPerObject.getData<int64_t>(channel_names::globalIds);
    auto motions = relativeOV_->local()->dataPerObject.getData<RigidMotion>(channel_names::motions);

    ids    ->downloadFromDevice(stream, ContainersSynch::Asynch);
    motions->downloadFromDevice(stream, ContainersSynch::Synch);

    for (size_t i = 0; i < ids->size(); i++)
    {
        if ((*ids)[i] == relativeID_)
        {
            real3 params[2] = { make_real3( (*motions)[i].r   ),
                                make_real3( (*motions)[i].vel ) };

            params[0] = getState()->domain.local2global(params[0]);

            for (int r = 0; r < nranks_; r++)
                MPI_Send(&params, NCOMPONENTS, getMPIFloatType<real>(), r, TAG, comm_);

            break;
        }
    }

    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );

    relativeParams[0] = getState()->domain.global2local(relativeParams[0]);

    for (auto& pv : pvs_)
        sampleOnePv(relativeParams[0], pv, stream);

    accumulateSampledAndClear(stream);

    averageRelativeVelocity_ += relativeParams[1];

    nSamples_++;
}


void AverageRelative3D::extractLocalBlock()
{
    static const double scale_by_density = -1.0;

    auto oneChannel = [this] (const PinnedBuffer<double>& channel, Average3D::ChannelType type, double scale, std::vector<double>& dest) {

        MPI_Check( MPI_Allreduce(MPI_IN_PLACE, channel.hostPtr(), channel.size(), MPI_DOUBLE, MPI_SUM, comm_) );

        const int ncomponents = this->getNcomponents(type);

        const int3 globalResolution = resolution_ * nranks3D_;

        double factor;
        int dstId = 0;
        for (int k = rank3D_.z * resolution_.z; k < (rank3D_.z+1) * resolution_.z; ++k)
        {
            for (int j = rank3D_.y * resolution_.y; j < (rank3D_.y+1) * resolution_.y; ++j)
            {
                for (int i = rank3D_.x * resolution_.x; i < (rank3D_.x+1) * resolution_.x; ++i)
                {
                    const int scalId = (k*globalResolution.y*globalResolution.x + j*globalResolution.x + i);
                    int srcId = ncomponents * scalId;
                    for (int c = 0; c < ncomponents; ++c)
                    {
                        if (scale == scale_by_density) factor = 1.0_r / accumulatedNumberDensity_[scalId];
                        else                           factor = scale;

                        dest[dstId++] = channel[srcId] * factor;
                        srcId++;
                    }
                }
            }
        }
    };

    // Order is important! Density comes first
    oneChannel(accumulatedNumberDensity_, Average3D::ChannelType::Scalar, 1.0 / (nSamples_ * binSize_.x*binSize_.y*binSize_.z), localNumberDensity_);

    for (int i = 0; i < channelsInfo_.n; ++i)
        oneChannel(accumulatedAverage_[i], channelsInfo_.types[i], scale_by_density, localChannels_[i]);
}

void AverageRelative3D::serializeAndSend(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    for (int i = 0; i < channelsInfo_.n; ++i)
    {
        auto& data = accumulatedAverage_[i];

        if (channelsInfo_.names[i] == channel_names::velocities)
        {
            constexpr int nthreads = 128;
            const int numVec3 = data.size() / 3;

            SAFE_KERNEL_LAUNCH
                (sampling_helpers_kernels::correctVelocity,
                 getNblocks(numVec3, nthreads), nthreads, 0, stream,
                 numVec3, reinterpret_cast<double3*> (data.devPtr()),
                 accumulatedNumberDensity_.devPtr(), averageRelativeVelocity_ / static_cast<real>(nSamples_));

            averageRelativeVelocity_ = make_real3(0.0_r);
        }
    }

    accumulatedNumberDensity_.downloadFromDevice(stream, ContainersSynch::Asynch);
    accumulatedNumberDensity_.clearDevice(stream);

    for (auto& data : accumulatedAverage_)
    {
        data.downloadFromDevice(stream, ContainersSynch::Asynch);
        data.clearDevice(stream);
    }

    CUDA_Check( cudaStreamSynchronize(stream) );

    extractLocalBlock();
    nSamples_ = 0;

    MirState::StepType timeStamp = getTimeStamp(getState(), dumpEvery_) - 1; // -1 to start from 0

    debug2("Plugin '%s' is now packing the data", getCName());
    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, timeStamp, localNumberDensity_, localChannels_);
    _send(sendBuffer_);
}

} // namespace mirheo
