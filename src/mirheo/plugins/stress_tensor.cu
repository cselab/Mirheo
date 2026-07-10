// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "stress_tensor.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

//#include <mirheo/core/datatypes.h>
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


namespace stress_tensor_kernels
{
__global__ void totalStress(PVview view, const real mass, stress_tensor_plugin::ReductionType *StressTensor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    stress_tensor_plugin::ReductionType P{};

    const int N = view.size;

    if (tid < N){
        const real4 r = view.positions[tid];
        const real4 v = view.velocities[tid];
        const real4 f = view.forces[tid];

        const real mass_xy = mass*v.x*v.y;
        const real mass_xz = mass*v.x*v.z;
        const real mass_yz = mass*v.y*v.z;

        P.p[0] = mass*v.x*v.x + r.x*f.x;
        P.p[1] = mass_xy + r.x*f.y;
        P.p[2] = mass_xz + r.x*f.z;
        P.p[3] = mass_xy + r.y*f.x;
        P.p[4] = mass*v.y*v.y + r.y*f.y;
        P.p[5] = mass_yz + r.y*f.z;
        P.p[6] = mass_xz + r.z*f.x;
        P.p[7] = mass_yz + r.z*f.y;
        P.p[8] = mass*v.z*v.z + r.z*f.z;
    }

    auto sum = [](real a, real b) { return a+b; };

    P.p[0] = warpReduce(P.p[0], sum);
    P.p[1] = warpReduce(P.p[1], sum);
    P.p[2] = warpReduce(P.p[2], sum);
    P.p[3] = warpReduce(P.p[3], sum);
    P.p[4] = warpReduce(P.p[4], sum);
    P.p[5] = warpReduce(P.p[5], sum);
    P.p[6] = warpReduce(P.p[6], sum);
    P.p[7] = warpReduce(P.p[7], sum);
    P.p[8] = warpReduce(P.p[8], sum);

    if (laneId() == 0){
        atomicAdd(&StressTensor->p[0], P.p[0]);
        atomicAdd(&StressTensor->p[1], P.p[1]);
        atomicAdd(&StressTensor->p[2], P.p[2]);
        atomicAdd(&StressTensor->p[3], P.p[3]);
        atomicAdd(&StressTensor->p[4], P.p[4]);
        atomicAdd(&StressTensor->p[5], P.p[5]);
        atomicAdd(&StressTensor->p[6], P.p[6]);
        atomicAdd(&StressTensor->p[7], P.p[7]);
        atomicAdd(&StressTensor->p[8], P.p[8]);
    }
}
} // namespace stress_tensor_kernels

StressTensorPlugin::StressTensorPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    dumpEvery_(dumpEvery)
{}

StressTensorPlugin::~StressTensorPlugin() = default;

void StressTensorPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    info("Plugin %s initialized for the following particle vector: %s", getCName(), pvName_.c_str());
}

void StressTensorPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, pvName_);
    _send(sendBuffer_);
}

//manually delete the halo forces (Not sure if important)
void StressTensorPlugin::beforeForces(cudaStream_t stream){
    pv_->halo()->forces().clearDevice(stream);
}

void StressTensorPlugin::beforeIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    PVview viewLocal(pv_, pv_->local());

    localStressTensor_.clear(stream);

    constexpr int nthreads = 128;
    const int nblocksL = getNblocks(viewLocal.size, nthreads);

    SAFE_KERNEL_LAUNCH(
        stress_tensor_kernels::totalStress,
        nblocksL, nthreads, 0, stream,
        viewLocal, pv_->getMassPerParticle(), localStressTensor_.devPtr() );

    //also summing up the halo forces (halo forces have to be saved as well!)
    PVview viewHalo(pv_, pv_->halo());
    const int nblocksH = getNblocks(viewHalo.size, nthreads);
    SAFE_KERNEL_LAUNCH(
        stress_tensor_kernels::totalStress,
        nblocksH, nthreads, 0, stream,
        viewHalo, pv_->getMassPerParticle(), localStressTensor_.devPtr() );

    

    localStressTensor_.downloadFromDevice(stream, ContainersSynch::Synch);

    savedTime_ = getState()->currentTime;
    needToSend_ = true;
}

void StressTensorPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_) return;

    debug2("Plugin %s is sending now data", getCName());

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, localStressTensor_[0]);
    _send(sendBuffer_);

    needToSend_ = false;
}

//=================================================================================

StressTensorDumper::StressTensorDumper(std::string name, std::string mask, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path)),
    fname_(name),
    mask_(mask)
{}

void StressTensorDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);

    if (mask_.size() != 9) {
        die("mask_.size() != 9, got %lu, example mask = 011101010 -> Pxy,Pxz,Pyx,Pyz,Pzy\n", mask_.size());
    }

    comment_ += "time";
    for(int i = 0; i < 9; ++i){
        bool_mask[i] = bool(mask_[i] - '0');
        if(bool_mask[i]){
            comment_.append(",");
            comment_.append(stress_label[i]);
        }
    }
    comment_ += "\n";
}

void StressTensorDumper::handshake()
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
        fprintf(fdump_.get(), comment_.c_str());
    }
}

void StressTensorDumper::deserialize()
{
    MirState::TimeType curTime;
    stress_tensor_plugin::ReductionType localStress, totalStress;

    SimpleSerializer::deserialize(data_, curTime, localStress);

    if (!activated_) return;

    const auto dataType = getMPIFloatType<real>();

    static_assert(sizeof(totalStress) == 9 * sizeof(real), "unexpected sizeof(Stress)");

    MPI_Check( MPI_Reduce(&localStress, &totalStress, 9, dataType, MPI_SUM, 0, comm_) );
    
    fprintf(fdump_.get(), "%.8e", curTime);
    for(int i = 0; i < 9; ++i){
        if(bool_mask[i]){
            fprintf(fdump_.get(), ",%.6e", totalStress.p[i]);
        }
    }
    fprintf(fdump_.get(), "\n");
}

} // namespace mirheo
