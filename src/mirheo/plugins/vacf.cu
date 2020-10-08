// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "vacf.h"
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

namespace vacf_kernels
{
__global__ void computeLocalVacf(PVview view, const real4 *velocities0, vacf_plugin::ReductionType *vacfSum)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    vacf_plugin::ReductionType vacf = 0;

    if (tid < view.size) {
        const auto v0 = Real3_int(velocities0[tid]).v;
        const auto v = Real3_int(view.readVelocity(tid)).v;

        vacf = dot(v, v0) / 3.0;
    }

    vacf = warpReduce(vacf, [](auto a, auto b) { return a+b; });

    if (laneId() == 0)
        atomicAdd(vacfSum, vacf);
}
} // namespace vacf_kernels

VacfPlugin::VacfPlugin(const MirState *state, std::string name, std::string pvName,
                       MirState::TimeType startTime, MirState::TimeType endTime, int dumpEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    startTime_(startTime),
    endTime_(endTime),
    dumpEvery_(dumpEvery)
{}

VacfPlugin::~VacfPlugin() = default;

void VacfPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    v0Channel_ = this->getName() + "_v0";

    pv_ = simulation->getPVbyNameOrDie(pvName_);
    pv_->requireDataPerParticle<real4>(v0Channel_, DataManager::PersistenceMode::Active);

    info("Plugin %s initialized for the following particle vector: %s", getCName(), pvName_.c_str());
}

void VacfPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, pvName_);
    _send(sendBuffer_);
}

void VacfPlugin::afterIntegration(cudaStream_t stream)
{
    const auto currentTime = getState()->currentTime;
    const auto currentStep = getState()->currentStep;

    if (currentTime < startTime_ || currentTime > endTime_)
        return;

    auto v0 = pv_->local()->dataPerParticle.getData<real4>(v0Channel_);

    if (startStep_ < 0)
    {
        v0->copy(pv_->local()->velocities());
        startStep_ = currentStep;
    }

    if ((currentStep - startStep_) % dumpEvery_ != 0)
        return;

    PVview view(pv_, pv_->local());
    localVacf_.clear(stream);

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    SAFE_KERNEL_LAUNCH(
        vacf_kernels::computeLocalVacf,
        nblocks, nthreads, 0, stream,
        view, v0->devPtr(), localVacf_.devPtr() );

    nparticles_ = view.size;
    localVacf_.downloadFromDevice(stream, ContainersSynch::Synch);

    savedTime_ = getState()->currentTime - startTime_;
    needToSend_ = true;
}

void VacfPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_)
        return;

    debug2("Plugin %s is sending now data", getCName());

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, localVacf_[0], nparticles_);
    _send(sendBuffer_);

    needToSend_ = false;
}

//=================================================================================

VacfDumper::VacfDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path))
{}

void VacfDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void VacfDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data_, pvName);

    if (activated_ && fdump_.get() == nullptr)
    {
        auto fname = joinPaths(path_, setExtensionOrDie(pvName, "csv"));
        auto status = fdump_.open(fname, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", fname.c_str());
        fprintf(fdump_.get(), "time,vacf\n");
    }
}

void VacfDumper::deserialize()
{
    MirState::TimeType curTime;
    vacf_plugin::ReductionType localVacf, totalVacf;
    long localNumParticles, totalNumParticles;

    SimpleSerializer::deserialize(data_, curTime, localVacf, localNumParticles);

    if (!activated_) return;

    const auto dataType = getMPIFloatType<vacf_plugin::ReductionType>();
    MPI_Check( MPI_Reduce(&localVacf, &totalVacf, 1, dataType, MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(&localNumParticles, &totalNumParticles, 1, MPI_LONG, MPI_SUM, 0, comm_) );

    fprintf(fdump_.get(), "%g,%.6e\n", curTime, totalVacf / totalNumParticles);
}

} // namespace mirheo
