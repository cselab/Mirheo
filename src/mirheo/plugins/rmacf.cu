// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "rmacf.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/chain_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>

namespace mirheo {
namespace rmacf_kernels {

__global__ void computeRouseMode(OVview view, int p, real3 *rouse_modes)
{
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = view.objSize;

    if (cid < view.nObjects)
    {
        real3 Xp = make_real3(0.0_r);

        for (int i = 0; i < nb; ++i)
        {
            const int pid = cid * nb + i;
            const auto r = Real3_int(view.readPosition(pid)).v;
            Xp += math::cos(p * M_PI/nb  *  ((i+1)-0.5_r)) * r;
        }
        Xp *= math::sqrt(2.0_r / nb);

        rouse_modes[cid] = Xp;
    }
}


__global__ void computeLocalRmacf(OVview view, int p, const real3 *rouse_modes_t0, rmacf_plugin::ReductionType *rmacfSum)
{
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = view.objSize;

    rmacf_plugin::ReductionType rmacf = 0;

    if (cid < view.nObjects)
    {
        const real3 Xp0 = rouse_modes_t0[cid];

        real3 Xp = make_real3(0.0_r);

        for (int i = 0; i < nb; ++i)
        {
            const int pid = cid * nb + i;
            const auto r = Real3_int(view.readPosition(pid)).v;
            Xp += math::cos(p * M_PI/nb  *  ((i+1)-0.5_r)) * r;
        }

        Xp *= math::sqrt(2.0_r / nb);

        rmacf = dot(Xp, Xp0);
    }

    rmacf = warpReduce(rmacf, [](auto a, auto b) { return a+b; });

    if (laneId() == 0)
        atomicAdd(rmacfSum, rmacf);
}

} // namespace rmacf_kernels

RmacfPlugin::RmacfPlugin(const MirState *state, std::string name, std::string cvName,
                         MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery) :
    SimulationPlugin(state, name),
    cvName_(cvName),
    startTime_(startTime),
    endTime_(endTime),
    dumpEvery_(dumpEvery)
{}

RmacfPlugin::~RmacfPlugin() = default;

void RmacfPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    cv_ = dynamic_cast<ChainVector*>(simulation->getPVbyNameOrDie(cvName_));

    if (!cv_)
    {
        die("%s expected a ChainVector, got %s", getCName(), cvName_.c_str());
    }

    for (int p = 1; p < cv_->getObjectSize(); ++p)
    {
        cv_->requireDataPerObject<real3>(_channelName(p), DataManager::PersistenceMode::Active);
        localRmacf_.emplace_back(1);
    }

    info("Plugin %s initialized for the following chain vector: %s", getCName(), cvName_.c_str());
}

void RmacfPlugin::handshake()
{
    const int numModes = cv_->getObjectSize() - 1;
    SimpleSerializer::serialize(sendBuffer_, cvName_, numModes);
    _send(sendBuffer_);
}

void RmacfPlugin::afterIntegration(cudaStream_t stream)
{
    const auto currentTime = getState()->currentTime;
    const auto currentStep = getState()->currentStep;

    if (currentTime < startTime_ || currentTime > endTime_)
        return;

    OVview view(cv_, cv_->local());

    if (startStep_ < 0)
    {
        for (int p = 1; p < cv_->getObjectSize(); ++p)
        {
            PinnedBuffer<real3> *Xp = cv_->local()->dataPerObject.getData<real3>(_channelName(p));
            constexpr int nthreads = 128;
            const int nblocks = getNblocks(view.nObjects, nthreads);

            SAFE_KERNEL_LAUNCH(
                rmacf_kernels::computeRouseMode,
                nblocks, nthreads, 0, stream,
                view, p, Xp->devPtr());
        }

        startStep_ = currentStep;
    }

    if ((currentStep - startStep_) % dumpEvery_ != 0)
        return;

    for (int p = 1; p < cv_->getObjectSize(); ++p)
    {
        localRmacf_[p-1].clear(stream);
        PinnedBuffer<real3> *Xp = cv_->local()->dataPerObject.getData<real3>(_channelName(p));

        constexpr int nthreads = 128;
        const int nblocks = getNblocks(view.nObjects, nthreads);

        SAFE_KERNEL_LAUNCH(
            rmacf_kernels::computeLocalRmacf,
            nblocks, nthreads, 0, stream,
            view, p, Xp->devPtr(), localRmacf_[p-1].devPtr() );

        localRmacf_[p-1].downloadFromDevice(stream, ContainersSynch::Asynch);
    }

    CUDA_Check( cudaStreamSynchronize(stream) );

    nchains_ = view.nObjects;

    savedTime_ = getState()->currentTime - startTime_;
    needToSend_ = true;
}

void RmacfPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_)
        return;

    std::vector<rmacf_plugin::ReductionType> rmacf(localRmacf_.size());

    for (size_t i = 0; i < rmacf.size(); ++i)
        rmacf[i] = localRmacf_[i][0];

    debug2("Plugin %s is now sending data", getCName());

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, rmacf, nchains_);
    _send(sendBuffer_);

    needToSend_ = false;
}

std::string RmacfPlugin::_channelName(int p) const
{
    return this->getName() + "_rm" + std::to_string(p);
}

//=================================================================================

RmacfDumper::RmacfDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

void RmacfDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void RmacfDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string cvName;
    SimpleSerializer::deserialize(data_, cvName, numModes_);

    if (activated_ && fdump_.get() == nullptr)
    {
        auto fname = joinPaths(path_, setExtensionOrDie(cvName, "csv"));
        auto status = fdump_.open(fname, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", fname.c_str());

        fprintf(fdump_.get(), "time");
        for (int i = 0; i < numModes_; ++i)
        {
            const int p = i+1;
            fprintf(fdump_.get(), ",rmacf%d", p);
        }
        fprintf(fdump_.get(), "\n");
    }
}

void RmacfDumper::deserialize()
{
    MirState::TimeType curTime;
    std::vector<rmacf_plugin::ReductionType> localRmacf(numModes_, 0.0);
    auto totalRmacf = localRmacf;
    long localNumChains{0}, totalNumChains{0};

    SimpleSerializer::deserialize(data_, curTime, localRmacf, localNumChains);

    if (!activated_) return;

    const auto dataType = getMPIFloatType<rmacf_plugin::ReductionType>();

    MPI_Check( MPI_Reduce(localRmacf.data(), totalRmacf.data(), numModes_, dataType, MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(&localNumChains, &totalNumChains, 1, MPI_LONG, MPI_SUM, 0, comm_) );

    if (rank_ == 0)
    {
        fprintf(fdump_.get(), "%g", curTime);

        for (int i = 0; i < numModes_; ++i)
        {
            const auto val = totalRmacf[i] / totalNumChains;
            fprintf(fdump_.get(), ",%g", val);
        }
        fprintf(fdump_.get(), "\n");
    }
}

} // namespace mirheo
