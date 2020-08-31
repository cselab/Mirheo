// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "msd.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>

namespace mirheo
{

namespace msd_kernels
{
__global__ void initPositionsAndDisplacements(PVview view, real4 *prevPositions, real4 *totalDisplacements)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= view.size)
        return;

    const real4 r = view.readPosition(i);

    totalDisplacements[i] = make_real4(0.0_r);
    prevPositions[i] = r;
}

__global__ void updatePositionsAndDisplacements(PVview view, real4 *prevPositions, real4 *totalDisplacements)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= view.size)
        return;

    const real4 r0 = prevPositions[i];
    const real4 r1 = view.readPosition(i);

    totalDisplacements[i] += r1 - r0;
    prevPositions[i] = r1;
}

__global__ void computeLocalMsd(int n, const real4 *totalDisplacements, msd_plugin::ReductionType *dispSum)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    msd_plugin::ReductionType disp = 0;

    if (i < n) {
        const auto d = Real3_int(totalDisplacements[i]).v;
        disp = dot(d, d);
    }

    disp = warpReduce(disp, [](auto a, auto b) { return a+b; });

    if (laneId() == 0)
        atomicAdd(dispSum, disp);
}
} // namespace msd_kernels

MsdPlugin::MsdPlugin(const MirState *state, std::string name, std::string pvName,
                       MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    startTime_(startTime),
    endTime_(endTime),
    dumpEvery_(dumpEvery)
{}

MsdPlugin::~MsdPlugin() = default;

void MsdPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    previousPositionChannelName_  = this->getName() + "_xprev";
    totalDisplacementChannelName_ = this->getName() + "_disp";

    pv_ = simulation->getPVbyNameOrDie(pvName_);
    pv_->requireDataPerParticle<real4>(previousPositionChannelName_, DataManager::PersistenceMode::Active, DataManager::ShiftMode::Active);
    pv_->requireDataPerParticle<real4>(totalDisplacementChannelName_, DataManager::PersistenceMode::Active);

    info("Plugin %s initialized for the following particle vector: %s", getCName(), pvName_.c_str());
}

void MsdPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, pvName_);
    _send(sendBuffer_);
}

void MsdPlugin::afterIntegration(cudaStream_t stream)
{
    const auto currentTime = getState()->currentTime;
    const auto currentStep = getState()->currentStep;

    if (currentTime < startTime_ || currentTime > endTime_)
        return;

    PVview view(pv_, pv_->local());

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    auto xPrev = pv_->local()->dataPerParticle.getData<real4>(previousPositionChannelName_);
    auto disp = pv_->local()->dataPerParticle.getData<real4>(totalDisplacementChannelName_);

    if (startStep_ < 0)
    {
        SAFE_KERNEL_LAUNCH(
            msd_kernels::initPositionsAndDisplacements,
            nblocks, nthreads, 0, stream,
            view, xPrev->devPtr(), disp->devPtr());

        startStep_ = currentStep;
    }

    if ((currentStep - startStep_) % dumpEvery_ != 0)
        return;


    SAFE_KERNEL_LAUNCH(
        msd_kernels::updatePositionsAndDisplacements,
        nblocks, nthreads, 0, stream,
        view, xPrev->devPtr(), disp->devPtr());

    localMsd_.clear(stream);

    SAFE_KERNEL_LAUNCH(
        msd_kernels::computeLocalMsd,
        nblocks, nthreads, 0, stream,
        view.size, disp->devPtr(), localMsd_.devPtr() );

    nparticles_ = view.size;
    localMsd_.downloadFromDevice(stream, ContainersSynch::Synch);

    savedTime_ = getState()->currentTime - startTime_;
    needToSend_ = true;
}

void MsdPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_)
        return;

    debug2("Plugin %s is sending now data", getCName());

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, localMsd_[0], nparticles_);
    _send(sendBuffer_);

    needToSend_ = false;
}

//=================================================================================

MsdDumper::MsdDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

void MsdDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void MsdDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data_, pvName);

    if (activated_)
    {
        auto fname = joinPaths(path_, setExtensionOrDie(pvName, "csv"));
        auto status = fdump_.open(fname, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", fname.c_str());
        fprintf(fdump_.get(), "time,msd\n");
    }
}

void MsdDumper::deserialize()
{
    MirState::TimeType curTime;
    msd_plugin::ReductionType localMsd, totalMsd;
    long localNumParticles, totalNumParticles;

    SimpleSerializer::deserialize(data_, curTime, localMsd, localNumParticles);

    if (!activated_) return;

    const auto dataType = getMPIFloatType<msd_plugin::ReductionType>();
    MPI_Check( MPI_Reduce(&localMsd, &totalMsd, 1, dataType, MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(&localNumParticles, &totalNumParticles, 1, MPI_LONG, MPI_SUM, 0, comm_) );

    fprintf(fdump_.get(), "%g,%.6e\n", curTime, totalMsd / totalNumParticles);
}

} // namespace mirheo
