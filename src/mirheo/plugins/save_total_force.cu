// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "save_total_force.h"
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

namespace total_force_kernels
{
__global__ void totalForce(PVview view, total_force_saver_plugin::ReductionType *force)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    total_force_saver_plugin::ReductionType F;

    if (tid < view.size){
        F.x = view.forces[tid].x;
        F.y = view.forces[tid].y;
        F.z = view.forces[tid].z;
    }

    auto sum = [](real a, real b) { return a+b; };

    F.x = warpReduce(F.x, sum);
    F.y = warpReduce(F.y, sum);
    F.z = warpReduce(F.z, sum);

    if (laneId() == 0){
        atomicAdd(&force->x, F.x);
        atomicAdd(&force->y, F.y);
        atomicAdd(&force->z, F.z);
    }
}
} // namespace total_force_kernels

TotalForceSaverPlugin::TotalForceSaverPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    dumpEvery_(dumpEvery)
{}

TotalForceSaverPlugin::~TotalForceSaverPlugin() = default;

void TotalForceSaverPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    info("Plugin %s initialized for the following particle vector: %s", getCName(), pvName_.c_str());
}

void TotalForceSaverPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, pvName_);
    _send(sendBuffer_);
}

void TotalForceSaverPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    PVview view(pv_, pv_->local());

    localForce_.clear(stream);

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    SAFE_KERNEL_LAUNCH(
        total_force_kernels::totalForce,
        nblocks, nthreads, 0, stream,
        view, localForce_.devPtr() );

        localForce_.downloadFromDevice(stream, ContainersSynch::Synch);

    savedTime_ = getState()->currentTime;
    needToSend_ = true;
}

void TotalForceSaverPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_) return;

    debug2("Plugin %s is sending now data", getCName());

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, localForce_[0]);
    _send(sendBuffer_);

    needToSend_ = false;
}

//=================================================================================

TotalForceSaverDumper::TotalForceSaverDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path)),
    fname_(name)
{}

void TotalForceSaverDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void TotalForceSaverDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data_, pvName);

    if (activated_ && fdump_.get() == nullptr)
    {
        //auto fname = joinPaths(path_, setExtensionOrDie(pvName, "csv"));
        auto fname = joinPaths(path_, setExtensionOrDie(fname_, "csv"));
        auto status = fdump_.open(fname, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", fname.c_str());
        fprintf(fdump_.get(), "time,force_x,force_y,force_z\n");
    }
}

void TotalForceSaverDumper::deserialize()
{
    MirState::TimeType curTime;
    total_force_saver_plugin::ReductionType localForce, totalForce;

    SimpleSerializer::deserialize(data_, curTime, localForce);

    if (!activated_) return;

    const auto dataType = getMPIFloatType<real>();

    static_assert(sizeof(totalForce) == 3 * sizeof(real), "unexpected sizeof(Stress)");

    MPI_Check( MPI_Reduce(&localForce, &totalForce, 3, dataType, MPI_SUM, 0, comm_) );


    fprintf(fdump_.get(), "%g,%.6e,%.6e,%.6e\n", curTime, totalForce.x, totalForce.y, totalForce.z);
}

} // namespace mirheo
